import json
import struct
import tempfile
import unittest

from pathlib import Path

from dk64_lib.data_types.geometry import (
    ACTOR_BONE_COUNT,
    ACTOR_DISPLAY_LIST_COUNT,
    ActorModelData,
    PropModelData,
    StageModelData,
)
from dk64_lib.f3dex2.display_list import (
    G_CULL_BACK,
    G_CULL_FRONT,
    G_TEXTURE_GEN,
    G_TEXTURE_GEN_LINEAR,
    ModelBone,
    ModelDecoder,
    read_bones,
)


def _words(word0: int, word1: int) -> bytes:
    return word0.to_bytes(4, "big") + word1.to_bytes(4, "big")


def _vertex(x: int, y: int, z: int) -> bytes:
    return struct.pack(">hhhHhhBBBB", x, y, z, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF)


def _bone(parent: int, index: int, x: float, y: float, z: float, channel: int = 0) -> bytes:
    return struct.pack(">BBBBfff", parent, index, channel, 0, x, y, z)


def _triangle_display_list(vertex_address: int) -> bytes:
    """Load three vertices from a segment address and draw them."""
    return b"".join(
        (
            _words(0x01003006, vertex_address),
            _words(0x05000204, 0x00000000),
            _words(0xDF000000, 0x00000000),
        )
    )


class ReadBonesTest(unittest.TestCase):
    def test_world_origins_accumulate_along_the_chain(self):
        data = _bone(0xFF, 0, 1.0, 0.0, 0.0) + _bone(0, 1, 0.0, 2.0, 0.0)
        bones = read_bones(data, 0, 2)

        self.assertEqual(bones[0].world, (1.0, 0.0, 0.0))
        self.assertEqual(bones[1].local, (0.0, 2.0, 0.0))
        self.assertEqual(bones[1].world, (1.0, 2.0, 0.0))

    def test_reads_the_animation_channel_apart_from_the_index(self):
        data = _bone(0xFF, 0, 0.0, 0.0, 0.0) + _bone(0, 1, 0.0, 1.0, 0.0, channel=0x16)
        bones = read_bones(data, 0, 2)

        self.assertEqual([bone.channel for bone in bones], [0, 0x16])


class ModelDecoderTest(unittest.TestCase):
    def test_moves_vertices_into_the_bind_pose_of_their_bone(self):
        vertices = _vertex(1, 2, 3) + _vertex(4, 5, 6) + _vertex(7, 8, 9)
        data = vertices + _words(0xDA380003, 0x04000040) + _triangle_display_list(0)
        bones = [
            ModelBone(0, 0xFF, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ModelBone(1, 0, (10.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
        ]

        decoder = ModelDecoder(data, lambda address: address, bones=bones)
        decoder.walk(len(vertices))
        group = decoder.mesh_groups[0]

        self.assertEqual([vertex.x for vertex in group.vertices], [11, 14, 17])
        self.assertEqual(group.vertex_bones, (1, 1, 1))

    def test_reads_linear_texture_gen_only_alongside_texture_gen(self):
        vertices = _vertex(0, 0, 0) + _vertex(1, 0, 0) + _vertex(0, 1, 0)
        for set_bits, expected in (
            (G_TEXTURE_GEN | G_TEXTURE_GEN_LINEAR, (True, True)),
            (G_TEXTURE_GEN, (True, False)),
            (G_TEXTURE_GEN_LINEAR, (False, False)),
        ):
            data = vertices + _words(0xD9FFFFFF, set_bits) + _triangle_display_list(0)

            decoder = ModelDecoder(data, lambda address: address)
            decoder.walk(len(vertices))
            render = decoder.mesh_groups[0].render

            self.assertEqual((render.texture_gen, render.texture_gen_linear), expected)

    def test_reads_which_faces_are_culled(self):
        vertices = _vertex(0, 0, 0) + _vertex(1, 0, 0) + _vertex(0, 1, 0)
        for set_bits, expected in (
            (0x00000000, (False, False)),
            (G_CULL_FRONT, (True, False)),
            (G_CULL_BACK, (False, True)),
            (G_CULL_FRONT | G_CULL_BACK, (True, True)),
        ):
            data = vertices + _words(0xD9FFFFFF, set_bits) + _triangle_display_list(0)

            decoder = ModelDecoder(data, lambda address: address)
            decoder.walk(len(vertices))
            group = decoder.mesh_groups[0]

            self.assertEqual((group.render.cull_front, group.render.cull_back), expected)
            self.assertEqual(group.double_sided, not any(expected))

    def test_keeps_walking_past_a_branch_the_game_fills_in_at_runtime(self):
        vertices = _vertex(0, 0, 0) + _vertex(1, 0, 0) + _vertex(0, 1, 0)
        data = vertices + _words(0xDE010000, 0x07000000) + _triangle_display_list(0)

        decoder = ModelDecoder(data, lambda address: None if address >> 24 else address)
        decoder.walk(len(vertices))

        self.assertEqual(sum(len(g.triangles) for g in decoder.mesh_groups), 1)

    def test_reads_the_highlight_call_until_a_tile_size_puts_the_tile_back(self):
        vertices = _vertex(0, 0, 0) + _vertex(1, 0, 0) + _vertex(0, 1, 0)
        draw = _words(0x01003006, 0x00000000) + _words(0x05000204, 0x00000000)
        data = (
            vertices
            + _words(0xDE000000, 0x05000000)
            + draw
            + _words(0xF2000000, 0x0007C07C)
            + draw
            + _words(0xDF000000, 0x00000000)
        )

        decoder = ModelDecoder(data, lambda address: None if address >> 24 else address)
        decoder.walk(len(vertices))

        self.assertEqual([group.render.hilite for group in decoder.mesh_groups], [True, False])


class ActorModelDataTest(unittest.TestCase):
    HEADER_SIZE = 0x28
    RAW_BASE = 0x10000000

    def _actor(self) -> ActorModelData:
        vertices = _vertex(1, 0, 0) + _vertex(0, 1, 0) + _vertex(0, 0, 1)
        display_list = _triangle_display_list(0x03000000)
        display_list_start = self.HEADER_SIZE + len(vertices)
        table_start = display_list_start + len(display_list)

        def raw(offset: int) -> int:
            return self.RAW_BASE + offset - self.HEADER_SIZE

        header = bytearray(self.HEADER_SIZE)
        struct.pack_into(">I", header, 0x00, self.RAW_BASE)
        struct.pack_into(">I", header, 0x04, raw(table_start))
        struct.pack_into(">I", header, 0x08, raw(table_start + 4))
        header[ACTOR_BONE_COUNT] = 1
        header[ACTOR_DISPLAY_LIST_COUNT] = 1
        raw_data = bytes(
            header
            + vertices
            + display_list
            + struct.pack(">I", raw(display_list_start))
            + _bone(0xFF, 0, 0.0, 0.0, 0.0)
        )
        return ActorModelData(
            raw_data=raw_data,
            offset=0,
            size=0,
            was_compressed=False,
            rom=None,
            index=0x12,
        )

    def test_decodes_its_display_list_table_and_skeleton(self):
        actor = self._actor()

        self.assertEqual(actor.name, "Candy")
        self.assertEqual(len(actor.bones), 1)
        self.assertEqual(sum(len(g.triangles) for g in actor.mesh_groups), 1)

    def test_decode_hands_back_the_mesh_groups_without_writing_files(self):
        actor = self._actor()
        groups, textures = actor.decode()

        self.assertEqual(sum(len(g.triangles) for g in groups), 1)
        self.assertEqual(textures, tuple())

    def test_writes_the_skeleton_into_the_glb_as_a_skin(self):
        glb = self._actor().create_textured_glb().data
        json_length = struct.unpack_from("<I", glb, 12)[0]
        gltf = json.loads(glb[20 : 20 + json_length])
        joints = gltf["skins"][0]["joints"]

        self.assertEqual([gltf["nodes"][joint]["name"] for joint in joints], ["bone_00"])
        for mesh in gltf["meshes"]:
            attributes = mesh["primitives"][0]["attributes"]
            self.assertEqual(
                gltf["accessors"][attributes["JOINTS_0"]]["count"],
                gltf["accessors"][attributes["POSITION"]]["count"],
            )

    def test_saves_every_supported_model_format(self):
        actor = self._actor()
        with tempfile.TemporaryDirectory() as tmpdir:
            for extension in ("obj", "dae", "gltf", "glb"):
                getattr(actor, f"save_to_{extension}")(f"actor.{extension}", tmpdir)
                self.assertTrue(Path(tmpdir, f"actor.{extension}").exists())

    def test_gltf_writes_one_file_and_leaves_the_rom_file_alone(self):
        actor = self._actor()
        with tempfile.TemporaryDirectory() as tmpdir:
            rom_file = Path(tmpdir, "actor.bin")
            rom_file.write_bytes(actor.raw_data)
            written_paths = actor.save_to_gltf("actor.gltf", tmpdir)

            self.assertEqual(written_paths, [Path(tmpdir, "actor.gltf")])
            self.assertEqual(rom_file.read_bytes(), actor.raw_data)
            gltf = json.loads(Path(tmpdir, "actor.gltf").read_text())
            self.assertTrue(gltf["buffers"][0]["uri"].startswith("data:"))


class PropModelDataTest(unittest.TestCase):
    def _prop(self, model_type: int = 1) -> PropModelData:
        vertices = _vertex(1, 0, 0) + _vertex(0, 1, 0) + _vertex(0, 0, 1)
        header = bytearray(0x70)
        header[0x0C:0x14] = b"pickups\x00"
        header[0x1C] = model_type
        struct.pack_into(">I", header, 0x40, len(header) + len(vertices))
        struct.pack_into(">I", header, 0x48, len(header))
        return PropModelData(
            raw_data=bytes(header + vertices + _triangle_display_list(0x08000000)),
            offset=0,
            size=0,
            was_compressed=False,
            rom=None,
            index=0x74,
        )

    def test_reads_its_category_and_geometry(self):
        prop = self._prop()

        self.assertEqual(prop.name, "pickups Golden Banana")
        self.assertEqual(sum(len(g.triangles) for g in prop.mesh_groups), 1)
        self.assertEqual(prop.bones, tuple())

    def test_rejects_props_that_are_not_geometry(self):
        with self.assertRaisesRegex(ValueError, "not geometry"):
            self._prop(model_type=2).mesh_groups


class StageModelDataTest(unittest.TestCase):
    def _stage(self, index: int) -> StageModelData:
        raw_data = bytes((0x00, 0x03, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00))
        return StageModelData(
            raw_data=raw_data,
            offset=0,
            size=len(raw_data),
            was_compressed=False,
            rom=None,
            index=index,
        )

    def test_is_named_from_the_map_list(self):
        self.assertEqual(self._stage(7).name, "Jungle Japes")

    def test_past_the_map_list_is_unknown(self):
        self.assertEqual(self._stage(0xFFFF).name, "unknown")


if __name__ == "__main__":
    unittest.main()
