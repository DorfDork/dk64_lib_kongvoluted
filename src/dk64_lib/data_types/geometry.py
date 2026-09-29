from __future__ import annotations

import re
import pathlib

from functools import cached_property
from typing import ClassVar

from numpy import array as numpy_array

try:
    from collada import Collada, source, material, geometry, scene
except ImportError:
    Collada = source = material = geometry = scene = None

from dk64_lib.binary_reader import BinaryReader
from dk64_lib.constants import ACTORS, MAPS, PROPS
from dk64_lib.data_types.base import BaseData
from dk64_lib.f3dex2.display_list import (
    DisplayList,
    DisplayListChunkData,
    DisplayListExpansion,
    _MeshGroup,
    ModelBone,
    ModelDecoder,
    create_display_lists,
    read_bones,
)
from dk64_lib.f3dex2.texture_export import (
    DecodedTexture,
    TextureAnimationFrames,
    TexturedDaeExport,
    TexturedDaeExporter,
    TexturedGlbExport,
    TexturedGltfExport,
    TexturedGltfExporter,
    TexturedObjExport,
    TexturedObjExporter,
    display_list_mesh_groups,
    save_textured_dae_export,
    save_textured_glb_export,
    save_textured_gltf_export,
    save_textured_obj_export,
)

POINTER_PATTERN = re.compile(b'\x00[\x00-\xFF]\x08\x00\x00\x00\x00\x00')

ACTOR_HEADER_SIZE = 0x28
ACTOR_RAW_BASE = 0x00
ACTOR_DISPLAY_LIST_TABLE = 0x04
ACTOR_BONE_TABLE = 0x08
ACTOR_TEXTURE_ANIMATIONS = 0x10
ACTOR_BONE_COUNT = 0x20
ACTOR_DISPLAY_LIST_COUNT = 0x21
ACTOR_DATA_SEGMENT = 0x03

PROP_CATEGORY = 0x0C
PROP_MODEL_TYPE = 0x1C
PROP_DISPLAY_LIST_1 = 0x40
PROP_DISPLAY_LIST_2 = 0x44
PROP_VERTEX_SEGMENT_BASE = 0x48
PROP_TEXTURE_ANIMATIONS = 0x6C
PROP_TEXTURE_ANIMATION_SIZE = 0x84
PROP_GEOMETRY_TYPE = 1
PROP_VERTEX_SEGMENT = 0x08


class TexturedModelData(BaseData):
    @property
    def mesh_groups(self) -> tuple[_MeshGroup, ...]:
        raise NotImplementedError

    @property
    def bones(self) -> tuple[ModelBone, ...]:
        return tuple()

    def _texture_tables(self) -> tuple[tuple[object, ...], ...]:
        if self.rom is None:
            return (tuple(),)
        return (self.rom.get_geometry_texture_data(), self.rom.get_animated_texture_data())

    def decode(self) -> tuple[tuple[_MeshGroup, ...], tuple[DecodedTexture, ...]]:
        return TexturedObjExporter(*self._texture_tables()).decode(self.mesh_groups)

    def create_textured_obj(
        self,
        mtl_filename: str = "model.mtl",
        texture_folder: str = "textures",
        include_textures: bool = True,
    ) -> TexturedObjExport:
        """Creates OBJ, MTL, and texture image data for this geometry."""
        return TexturedObjExporter(*self._texture_tables()).export(
            self.mesh_groups,
            mtl_filename=mtl_filename,
            texture_folder=texture_folder,
            include_textures=include_textures,
        )

    def create_textured_dae(
        self,
        texture_folder: str = "textures",
        animated_texture_frames: TextureAnimationFrames | None = None,
        animation_frame_duration: int = 4,
        include_textures: bool = True,
    ) -> TexturedDaeExport:
        """Creates DAE and texture image data for this geometry."""
        return TexturedDaeExporter(*self._texture_tables()).export(
            self.mesh_groups,
            texture_folder=texture_folder,
            animated_texture_frames=animated_texture_frames,
            animation_frame_duration=animation_frame_duration,
            include_textures=include_textures,
        )

    def create_textured_gltf(self, include_textures: bool = True) -> TexturedGltfExport:
        """Creates self-contained glTF data for this geometry."""
        return TexturedGltfExporter(*self._texture_tables()).export(
            self.mesh_groups,
            include_textures=include_textures,
        )

    def create_textured_glb(self, include_textures: bool = True) -> TexturedGlbExport:
        """Creates binary glTF data for this geometry."""
        return TexturedGltfExporter(*self._texture_tables()).export_glb(
            self.mesh_groups,
            include_textures=include_textures,
            skeleton=self.bones,
        )

    def save_to_obj(
        self,
        filename: str,
        folderpath: str = ".",
        include_textures: bool = True,
        texture_folder: str = "textures",
    ) -> list[pathlib.Path]:
        export = self.create_textured_obj(
            mtl_filename=pathlib.Path(filename).with_suffix(".mtl").name,
            texture_folder=texture_folder,
            include_textures=include_textures,
        )
        return save_textured_obj_export(export, filename, folderpath)

    def save_to_dae(
        self,
        filename: str,
        folderpath: str = ".",
        include_textures: bool = True,
        texture_folder: str = "textures",
        animated_texture_frames: TextureAnimationFrames | None = None,
        animation_frame_duration: int = 4,
    ) -> list[pathlib.Path]:
        export = self.create_textured_dae(
            texture_folder=texture_folder,
            animated_texture_frames=animated_texture_frames,
            animation_frame_duration=animation_frame_duration,
            include_textures=include_textures,
        )
        return save_textured_dae_export(export, filename, folderpath)

    def save_to_gltf(
        self,
        filename: str,
        folderpath: str = ".",
        include_textures: bool = True,
    ) -> list[pathlib.Path]:
        """Save geometry data to glTF format."""
        export = self.create_textured_gltf(include_textures=include_textures)
        return save_textured_gltf_export(export, filename, folderpath)

    def save_to_glb(
        self,
        filename: str,
        folderpath: str = ".",
        include_textures: bool = True,
    ) -> list[pathlib.Path]:
        """Save geometry data to binary glTF format."""
        export = self.create_textured_glb(include_textures=include_textures)
        return save_textured_glb_export(export, filename, folderpath)


class StageModelData(TexturedModelData):
    def __post_init__(self):
        self.data_type = "Geometry"
        self.is_pointer = False
        if POINTER_PATTERN.match(self.raw_data[:8]):
            self.is_pointer = True
        self.points_to = None

        self.dl_start = 0
        self.vert_start = 0
        self.vert_length = 0
        self.vert_chunk_start = 0
        self.vert_chunk_length = 0
        self.dl_expansion_start = 0
        if self.is_pointer:
            return

        reader = BinaryReader(self.raw_data)

        # Get the display list and vertex starts
        self.dl_start = reader.read_u32(0x34)
        self.vert_start = reader.read_u32(0x38)

        # ! I don't know what this is pointing to, but it signifies the end of the
        # ! vertex data which is importent
        _unknown_start = reader.read_u32(0x40)
        self.vert_length = _unknown_start - self.vert_start

        self.vert_chunk_start = reader.read_u32(0x68)

        # ! I don't know what this is pointing to, but it signifies the end of the
        # ! vertex chunk data which is importent
        _unknown_start = reader.read_u32(0x6C)
        self.vert_chunk_length = _unknown_start - self.vert_chunk_start

        self.dl_expansion_start = reader.read_u32(0x70)

    @property
    def name(self) -> str:
        return MAPS[self.index] if self.index < len(MAPS) else "unknown"

    @property
    def pointer(self):
        if self.is_pointer:
            return self.raw_data[1]
        return None

    @property
    def dl_expansions(self) -> list[DisplayListExpansion]:
        """Returns a list of the display list expansion data

        Returns:
            list[DisplayListExpansion]: Display list expansion data
        """
        if self.is_pointer:
            return list()
        ret_list = list()
        reader = BinaryReader(self.raw_data)
        expansion_count = reader.read_u32(self.dl_expansion_start)
        expansion_start = self.dl_expansion_start + 4
        for expansion_num in range(expansion_count):
            offset = expansion_start + expansion_num * 0x10
            ret_list.append(
                DisplayListExpansion.from_bytes(reader.read_at(offset, 0x10))
            )

        return ret_list

    @property
    def vertex_chunk_data(self) -> list[DisplayListChunkData]:
        """Returns a list of display list chunk data found in the geometry file

        Returns:
            list[DisplayListChunkData]: Display list chunk data
        """
        if self.is_pointer:
            return list()
        ret_list = list()
        for chunk_num in range(int(self.vert_chunk_length / 52)):
            chunk_start = self.vert_chunk_start + 52 * chunk_num
            chunk_end = self.vert_chunk_start + 52 * (chunk_num + 1)
            ret_list.append(
                DisplayListChunkData.from_bytes(self.raw_data[chunk_start:chunk_end])
            )
        return ret_list

    @property
    def display_lists(self) -> list[DisplayList]:
        """Generate and return a list of display lists inside of the geometry data

        Returns:
            list[DisplayList]: A list of display lists
        """
        if self.is_pointer:
            return list()

        raw_dl_data = self.raw_data[self.dl_start : self.vert_start]

        raw_vertex_data = self.raw_data[
            self.vert_start : self.vert_start + self.vert_length
        ]

        return create_display_lists(
            raw_dl_data,
            raw_vertex_data,
            self.vertex_chunk_data,
            expansions=self.dl_expansions,
        )

    @property
    def mesh_groups(self) -> tuple[_MeshGroup, ...]:
        return display_list_mesh_groups(self.display_lists)

    def create_obj(self) -> str:
        """Creates an obj file out of the geometry data

        Returns:
            str: Obj file data
        """
        obj_data = str()
        tri_offset = 1

        if self.is_pointer:
            return obj_data

        for dl_num, dl in enumerate(self.display_lists, 1):
            if dl.is_branched:
                continue

            obj_data += f"# Display List {dl_num}, Offset: {dl.offset}\n\n"

            for group_num, (verticies, triangles) in enumerate(
                zip(dl.verticies, dl.triangles), 1
            ):

                obj_data += f"# Vertex Group {group_num}\n\n"

                # Write vertecies to file
                for vertex in verticies:
                    obj_line = f"{vertex.to_obj_line()}\n"
                    obj_data += obj_line
                obj_data += "\n"

                obj_data += f"# Triangle Group {group_num}\n\n"

                # Write triangles/faces to file
                for tri in triangles:
                    obj_line = f"f {tri.v1 + tri_offset} {tri.v2 + tri_offset} {tri.v3 + tri_offset}\n"
                    obj_data += obj_line
                obj_data += "\n"

                # The triangle offset is used to globally identify the vertex due to
                # Display Lists reading them with local positions
                tri_offset += len(verticies)
        return obj_data

    def save_to_obj(
        self,
        filename: str,
        folderpath: str = ".",
        include_textures: bool = True,
        texture_folder: str = "textures",
    ) -> list[pathlib.Path]:
        """Save geometry data to obj format

        Args:
            filename (str): Name of obj file
            folderpath (str, optional): Folder path to save obj to. Defaults to ".".
            include_textures (bool, optional): Whether to export OBJ material and
                texture files alongside the OBJ. Defaults to True.
            texture_folder (str, optional): Folder for exported texture images
                when include_textures is True. Defaults to "textures".
        """
        if include_textures:
            return self.save_to_textured_obj(filename, folderpath, texture_folder)

        filepath = pathlib.Path(folderpath, filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.create_obj())
        return [filepath]

    def save_to_textured_obj(
        self,
        filename: str,
        folderpath: str = ".",
        texture_folder: str = "textures",
    ) -> list[pathlib.Path]:
        """Save OBJ, MTL, and texture PNG files for this geometry."""
        mtl_filename = pathlib.Path(filename).with_suffix(".mtl").name
        export = self.create_textured_obj(
            mtl_filename=mtl_filename,
            texture_folder=texture_folder,
        )
        return save_textured_obj_export(export, filename, folderpath)

    def create_dae(
        self,
        include_textures: bool = True,
        texture_folder: str = "textures",
        animated_texture_frames: TextureAnimationFrames | None = None,
        animation_frame_duration: int = 4,
    ) -> Collada:
        """Creates a dae file out of the geometry data

        Returns:
            Collada: dae file
        """
        if include_textures:
            return self.create_textured_dae(
                texture_folder,
                animated_texture_frames=animated_texture_frames,
                animation_frame_duration=animation_frame_duration,
            ).dae
        return self._create_geometry_only_dae()

    def _create_geometry_only_dae(self) -> Collada:
        """Creates a DAE file with geometry and vertex colors only."""
        if Collada is None:
            raise ImportError("DAE export needs pycollada")
        mesh = Collada()
        
        vertex_data = list()
        vertex_colour_data = list()
        triangle_data = list()
        tri_offset = 0
        
        for dl in self.display_lists:
            if dl.is_branched:
                continue

            for verticies, triangles in zip(dl.verticies, dl.triangles):

                # Write vertecies to file
                for vertex in verticies:
                    vertex_data.extend((vertex.x, vertex.y, vertex.z))
                    vertex_colour_data.extend((vertex.xr / 255, vertex.yg / 255, vertex.zb / 255, vertex.alpha / 255))

                # # Write triangles/faces to file
                for tri in triangles:
                    triangle_data.extend((tri.v1 + tri_offset, tri.v2 + tri_offset, tri.v3 + tri_offset))

                # # The triangle offset is used to globally identify the vertex due to
                # # Display Lists reading them with local positions
                tri_offset += len(verticies)
        
        # Create the vertices and vertex colours
        src_vertices = source.FloatSource("geo-vertices", numpy_array(vertex_data), ('X', 'Y', 'Z'))
        src_vertices_colour = source.FloatSource('vertex-colours', numpy_array(vertex_colour_data), ('R', 'G', 'B', 'A'))
        
        # The input list is used to define the vertices and colours 
        input_list = source.InputList()
        input_list.addInput(0, 'VERTEX', "#geo-vertices")
        input_list.addInput(0, 'COLOR', '#vertex-colours')

        # Create a geometry with the data points
        geom = geometry.Geometry(mesh, "geometry0", "map", [src_vertices, src_vertices_colour])
        mesh.geometries.append(geom)

        # Create a triangle set and append it to the geometry
        triset = geom.createTriangleSet(numpy_array(triangle_data), input_list, "vertex-material")
        geom.primitives.append(triset)
        
        # Create the phong effect and material using the vertex colours
        effect = material.Effect('vertex-effect', [], 'phong', diffuse=src_vertices_colour)
        mat = material.Material('vertex-material', 'vertex-material', effect)
        mesh.effects.append(effect)
        mesh.materials.append(mat)
        
        # Create and add the scene
        geomnode = scene.GeometryNode(geom, [])
        node = scene.Node("node0", children=[geomnode])
        myscene = scene.Scene("myscene", [node])
        mesh.scenes.append(myscene)
        mesh.scene = myscene
        
        return mesh
        
    def save_to_dae(
        self,
        filename: str,
        folderpath: str = ".",
        include_textures: bool = True,
        texture_folder: str = "textures",
        animated_texture_frames: TextureAnimationFrames | None = None,
        animation_frame_duration: int = 4,
    ) -> list[pathlib.Path]:
        """Save geometry data to dae format

        Args:
            filename (str): Name of dae file
            folderpath (str, optional): Folder path to save dae to. Defaults to ".".
            include_textures (bool, optional): Whether to export DAE materials and
                texture files alongside the DAE. Defaults to True.
            texture_folder (str, optional): Folder for exported texture images
                when include_textures is True. Defaults to "textures".
        """
        if include_textures:
            export = self.create_textured_dae(
                texture_folder=texture_folder,
                animated_texture_frames=animated_texture_frames,
                animation_frame_duration=animation_frame_duration,
            )
            return save_textured_dae_export(export, filename, folderpath)

        filepath = pathlib.Path(folderpath, filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._create_geometry_only_dae().write(filepath)
        return [filepath]


class ModelData(TexturedModelData):
    kind: ClassVar[str] = "model"

    def __post_init__(self):
        self.data_type = self.kind.capitalize()
        self._reader = BinaryReader(self.raw_data)

    @property
    def bones(self) -> tuple[ModelBone, ...]:
        return tuple(self._decoder.bones)

    @property
    def mesh_groups(self) -> tuple[_MeshGroup, ...]:
        return self._decoder.mesh_groups

    @cached_property
    def _decoder(self) -> ModelDecoder:
        decoder = self._decode()
        if not decoder.mesh_groups:
            raise ValueError("No triangles in the model's display lists")
        return decoder

    def _decode(self) -> ModelDecoder:
        raise NotImplementedError


class ActorModelData(ModelData):
    kind = "actor"

    @property
    def name(self) -> str:
        if self.index < len(ACTORS):
            return ACTORS[self.index]
        return self.kind

    @property
    def bone_count(self) -> int:
        return self._reader.read_u8(ACTOR_BONE_COUNT)

    def _decode(self) -> ModelDecoder:
        if len(self.raw_data) < ACTOR_HEADER_SIZE:
            raise ValueError("File is too small for an actor header")
        raw_base = self._reader.read_u32(ACTOR_RAW_BASE)

        def raw_to_offset(pointer: int) -> int:
            return pointer + ACTOR_HEADER_SIZE - raw_base

        def resolve_address(address: int) -> int | None:
            if address >> 24 == ACTOR_DATA_SEGMENT:
                return ACTOR_HEADER_SIZE + (address & 0xFFFFFF)
            offset = raw_to_offset(address)
            if address >> 24 == 0 or 0 <= offset < len(self.raw_data):
                return offset
            return None

        bones = self._bones(raw_to_offset)
        decoder = ModelDecoder(
            self.raw_data,
            resolve_address,
            bones=bones,
            texture_segments=self._texture_segments(raw_to_offset),
        )
        table = raw_to_offset(self._reader.read_u32(ACTOR_DISPLAY_LIST_TABLE))
        for entry in range(self._reader.read_u8(ACTOR_DISPLAY_LIST_COUNT)):
            offset = table + entry * 4
            if 0 <= offset and offset + 4 <= len(self.raw_data):
                decoder.walk(raw_to_offset(self._reader.read_u32(offset)))
        return decoder

    def _bones(self, raw_to_offset) -> list[ModelBone]:
        pointer = self._reader.read_u32(ACTOR_BONE_TABLE)
        if pointer == 0:
            return list()
        return read_bones(self.raw_data, raw_to_offset(pointer), self.bone_count)

    def _texture_segments(self, raw_to_offset) -> dict[int, int]:
        pointer = self._reader.read_u32(ACTOR_TEXTURE_ANIMATIONS)
        if pointer == 0:
            return dict()
        offset = raw_to_offset(pointer)
        if not 0 <= offset <= len(self.raw_data) - 2:
            return dict()

        count = self._reader.read_u16(offset)
        offset += 2
        segments = dict()
        for _ in range(count):
            if offset + 6 > len(self.raw_data):
                break
            frame_count = self._reader.read_u16(offset)
            segment = self._reader.read_u16(offset + 2)
            offset += 6
            if frame_count == 0 or offset + frame_count * 2 > len(self.raw_data):
                break
            segments[segment] = self._reader.read_u16(offset)
            offset += frame_count * 2
        return segments


class PropModelData(ModelData):
    kind = "prop"

    @property
    def name(self) -> str:
        name = PROPS[self.index] if self.index < len(PROPS) else ""
        return " ".join(part for part in (self.category, name) if part) or self.kind

    @property
    def category(self) -> str:
        raw_name = self.raw_data[PROP_CATEGORY:PROP_MODEL_TYPE].split(b"\0")[0]
        if raw_name and all(0x20 <= character < 0x7F for character in raw_name):
            return raw_name.decode("ascii")
        return ""

    @property
    def model_type(self) -> int:
        return self._reader.read_u8(PROP_MODEL_TYPE)

    def _decode(self) -> ModelDecoder:
        if len(self.raw_data) < PROP_TEXTURE_ANIMATIONS + 4:
            raise ValueError("File is too small for a prop header")
        if self.model_type != PROP_GEOMETRY_TYPE:
            raise ValueError(f"Type {self.model_type} props are not geometry")
        vertex_base = self._reader.read_u32(PROP_VERTEX_SEGMENT_BASE)

        def resolve_address(address: int) -> int | None:
            segment, offset = address >> 24, address & 0xFFFFFF
            if segment == PROP_VERTEX_SEGMENT:
                return vertex_base + offset
            if segment == 0:
                return offset
            return None

        decoder = ModelDecoder(
            self.raw_data,
            resolve_address,
            fallback_textures=self._animated_textures(),
        )
        for display_list in (PROP_DISPLAY_LIST_1, PROP_DISPLAY_LIST_2):
            offset = self._reader.read_u32(display_list)
            if offset:
                decoder.walk(offset)
        return decoder

    def _animated_textures(self) -> set[int]:
        offset = self._reader.read_u32(PROP_TEXTURE_ANIMATIONS)
        if offset + 4 > len(self.raw_data):
            return set()

        textures = set()
        for entry in range(self._reader.read_u32(offset)):
            start = offset + 4 + entry * PROP_TEXTURE_ANIMATION_SIZE
            if start + PROP_TEXTURE_ANIMATION_SIZE > len(self.raw_data):
                break
            textures.add(self._reader.read_u32(start))
        return textures
