# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild
import re
import sys

import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO
from enum import Enum

from CustomExceptions import InvalidDataException

if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 9):
    raise Exception(
        "Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

from . import dos_datetime


class Zip(KaitaiStruct):
    """ZIP is a popular archive file format, introduced in 1989 by Phil Katz
    and originally implemented in PKZIP utility by PKWARE.
    
    Thanks to solid support of it in most desktop environments and
    operating systems, and algorithms / specs availability in public
    domain, it quickly became tool of choice for implementing file
    containers.
    
    For example, Java .jar files, OpenDocument, Office Open XML, EPUB files
    are actually ZIP archives.
    
    .. seealso::
       Source - https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
    
    
    .. seealso::
       Source - https://users.cs.jmu.edu/buchhofp/forensics/formats/pkzip.html
    """

    class Compression(Enum):
        none = 0
        shrunk = 1
        reduced_1 = 2
        reduced_2 = 3
        reduced_3 = 4
        reduced_4 = 5
        imploded = 6
        deflated = 8
        enhanced_deflated = 9
        pkware_dcl_imploded = 10
        bzip2 = 12
        lzma = 14
        ibm_terse = 18
        ibm_lz77_z = 19
        zstandard = 93
        mp3 = 94
        xz = 95
        jpeg = 96
        wavpack = 97
        ppmd = 98
        aex_encryption_marker = 99

    class ExtraCodes(Enum):
        zip64 = 1
        av_info = 7
        os2 = 9
        ntfs = 10
        openvms = 12
        pkware_unix = 13
        file_stream_and_fork_descriptors = 14
        patch_descriptor = 15
        pkcs7 = 20
        x509_cert_id_and_signature_for_file = 21
        x509_cert_id_for_central_dir = 22
        strong_encryption_header = 23
        record_management_controls = 24
        pkcs7_enc_recip_cert_list = 25
        ibm_s390_uncomp = 101
        ibm_s390_comp = 102
        poszip_4690 = 18064
        extended_timestamp = 21589
        beos = 25922
        asi_unix = 30062
        infozip_unix = 30805
        infozip_unix_var_size = 30837
        aex_encryption = 39169
        apache_commons_compress = 41246
        microsoft_open_packaging_growth_hint = 41504
        sms_qdos = 64842

    def __init__(self, _io, _parent=None, _root=None, start_offset=0):
        self._io = _io
        self._parent = _parent
        self.start = self._io.pos() + start_offset  # + (_parent.start if _parent is not None else 0)
        self._root = _root if _root else self
        self._read(start_offset)
        self.end = self._io.pos() + start_offset

    def _read(self, start_offset):
        self.sections = []
        i = 0
        while not self._io.is_eof():
            self.sections.append(Zip.PkSection(self._io, self, self._root, start_offset=start_offset))
            i += 1

    class LocalFile(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, start_offset=0):
            self._io = _io
            self._parent = _parent
            self.start = self._io.pos() + start_offset  # + (_parent.start if _parent is not None else 0)
            self._root = _root if _root else self
            self._read(start_offset=start_offset)
            self.end = self._io.pos() + start_offset

        def _read(self, start_offset=0):
            self.header = Zip.LocalFileHeader(self._io, self, self._root, start_offset=start_offset)
            len_body_compressed_io_pos = self.header.start + 18
            def read_except(len_body):
                try:
                    exception_at = self._io.pos()
                    return self._io.read_bytes(len_body)
                except EOFError as ex:
                    allowed_bytes = int(ex.args[0].split("but only ")[1].split(" bytes available")[0])
                    raise InvalidDataException("/types/compressed_data/invalid", len_body_compressed_io_pos, expected=allowed_bytes)

            self.body = read_except(self.header.len_body_compressed)

    class DataDescriptor(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, start_offset=0):
            self._io = _io
            self._parent = _parent
            self.start = self._io.pos() + start_offset  # + (_parent.start if _parent is not None else 0)
            self._root = _root if _root else self
            self._read()
            self.end = self._io.pos() + start_offset

        def _read(self):
            self.crc32 = self._io.read_u4le()
            self.len_body_compressed = self._io.read_u4le()
            self.len_body_uncompressed = self._io.read_u4le()

    class ExtraField(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, start_offset=0):
            self._io = _io
            self._parent = _parent
            self.start = self._io.pos() + start_offset  # + (_parent.start if _parent is not None else 0)
            self._root = _root if _root else self
            self._read(start_offset=start_offset)
            self.end = self._io.pos() + start_offset

        def _read(self, start_offset=0):
            def read_except(len_body):
                try:
                    exception_at = self._io.pos()
                    return self._io.read_bytes(len_body)
                except EOFError as ex:
                    allowed_bytes = int(ex.args[0].split("but only ")[1].split(" bytes available")[0])
                    raise InvalidDataException("/types/raw_body/invalid", len_body_io_pos, expected=allowed_bytes)

            self.code = KaitaiStream.resolve_enum(Zip.ExtraCodes, self._io.read_u2le())
            len_body_io_pos = self._io.pos()
            self.len_body = self._io.read_u2le()
            _on = self.code
            if _on == Zip.ExtraCodes.ntfs:
                self._raw_body = read_except(self.len_body)
                _io__raw_body = KaitaiStream(BytesIO(self._raw_body))
                self.body = Zip.ExtraField.Ntfs(_io__raw_body, self, self._root,
                                                start_offset=start_offset + len_body_io_pos)
            elif _on == Zip.ExtraCodes.extended_timestamp:
                self._raw_body = read_except(self.len_body)
                _io__raw_body = KaitaiStream(BytesIO(self._raw_body))
                self.body = Zip.ExtraField.ExtendedTimestamp(_io__raw_body, self, self._root,
                                                             start_offset=start_offset + len_body_io_pos)
            elif _on == Zip.ExtraCodes.infozip_unix_var_size:
                self._raw_body = read_except(self.len_body)
                _io__raw_body = KaitaiStream(BytesIO(self._raw_body))
                self.body = Zip.ExtraField.InfozipUnixVarSize(_io__raw_body, self, self._root,
                                                              start_offset=start_offset + len_body_io_pos)
            else:
                self.body = read_except(self.len_body)

        class Ntfs(KaitaiStruct):
            """
            .. seealso::
               Source - https://github.com/LuaDist/zip/blob/b710806/proginfo/extrafld.txt#L191
            """

            def __init__(self, _io, _parent=None, _root=None, start_offset=0):
                self._io = _io
                self._parent = _parent
                self.start = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset
                self._root = _root if _root else self
                self._read(start_offset=start_offset)
                self.end = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset

            def _read(self, start_offset):
                self.reserved = self._io.read_u4le()
                self.attributes = []
                i = 0
                while not self._io.is_eof():
                    self.attributes.append(
                        Zip.ExtraField.Ntfs.Attribute(self._io, self, self._root, start_offset=start_offset))
                    i += 1

            class Attribute(KaitaiStruct):
                def __init__(self, _io, _parent=None, _root=None, start_offset=0):
                    self._io = _io
                    self._parent = _parent
                    self.start = self._io.pos() + start_offset  # + (_parent.start if _parent is not None else 0)
                    self._root = _root if _root else self
                    self._read(start_offset)
                    self.end = self._io.pos() + start_offset

                def _read(self, start_offset):
                    self.tag = self._io.read_u2le()
                    self.len_body = self._io.read_u2le()
                    _on = self.tag
                    if _on == 1:
                        self._raw_body = self._io.read_bytes(self.len_body)
                        _io__raw_body = KaitaiStream(BytesIO(self._raw_body))
                        self.body = Zip.ExtraField.Ntfs.Attribute1(_io__raw_body, self, self._root, start_offset)
                    else:
                        self.body = self._io.read_bytes(self.len_body)

            class Attribute1(KaitaiStruct):
                def __init__(self, _io, _parent=None, _root=None, start_offset=0):
                    self._io = _io
                    self._parent = _parent
                    self.start = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset
                    self._root = _root if _root else self
                    self._read()
                    self.end = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset

                def _read(self):
                    self.last_mod_time = self._io.read_u8le()
                    self.last_access_time = self._io.read_u8le()
                    self.creation_time = self._io.read_u8le()

        class ExtendedTimestamp(KaitaiStruct):
            """
            .. seealso::
               Source - https://github.com/LuaDist/zip/blob/b710806/proginfo/extrafld.txt#L817
            """

            def __init__(self, _io, _parent=None, _root=None, start_offset=0):
                self._io = _io
                self._parent = _parent
                self.start = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset
                self._root = _root if _root else self
                self._read(start_offset=start_offset)
                self.end = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset

            def _read(self, start_offset):
                self._raw_flags = self._io.read_bytes(1)
                _io__raw_flags = KaitaiStream(BytesIO(self._raw_flags))
                self.flags = Zip.ExtraField.ExtendedTimestamp.InfoFlags(_io__raw_flags, self, self._root, start_offset)
                if self.flags.has_mod_time:
                    self.mod_time = self._io.read_u4le()

                if self.flags.has_access_time:
                    try:
                        self.access_time = self._io.read_u4le()
                    except:
                        self.access_time = self.mod_time

                if self.flags.has_create_time:
                    try:
                        self.create_time = self._io.read_u4le()
                    except:
                        if hasattr(self, 'mod_time'):
                            self.create_time = self.mod_time
                        else:
                            self.create_time = self.access_time

            class InfoFlags(KaitaiStruct):
                def __init__(self, _io, _parent=None, _root=None, start_offset=0):
                    self._io = _io
                    self._parent = _parent
                    self.start = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset
                    self._root = _root if _root else self
                    self._read(start_offset=start_offset)
                    self.end = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset

                def _read(self, start_offset):
                    self.has_mod_time = self._io.read_bits_int_le(1) != 0
                    self.has_access_time = self._io.read_bits_int_le(1) != 0
                    self.has_create_time = self._io.read_bits_int_le(1) != 0
                    self.reserved = self._io.read_bits_int_le(5)

        class InfozipUnixVarSize(KaitaiStruct):
            """
            .. seealso::
               Source - https://github.com/LuaDist/zip/blob/b710806/proginfo/extrafld.txt#L1339
            """

            def __init__(self, _io, _parent=None, _root=None, start_offset=0):
                self._io = _io
                self._parent = _parent
                self.start = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset
                self._root = _root if _root else self
                self._read()
                self.end = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset

            def _read(self):
                self.version = self._io.read_u1()
                self.len_uid = self._io.read_u1()
                self.uid = self._io.read_bytes(self.len_uid)
                self.len_gid = self._io.read_u1()
                self.gid = self._io.read_bytes(self.len_gid)

    @staticmethod
    def find_signtures(input_bytes: bytes):
        try:
            res = input_bytes.find(
                re.search(rb'\x50\x4b\x03\x04|\x50\x4b\x01\x02|\x50\x4b\x05\x06|\x50\x4b\x07\x08', input_bytes).group(
                    0))
        except:
            res = None
        return res

    @staticmethod
    def find_first_difference(input_1, input_2):
        for i in range(min(len(input_1), len(input_2))):
            if input_1[i] != input_2[i]:
                return i
        return None

    class CentralDirEntry(KaitaiStruct):
        """
        .. seealso::
           - 4.3.12 - https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
        """

        def __init__(self, _io, _parent=None, _root=None, start_offset=0):
            self.start_offset = start_offset
            self._io = _io
            self._parent = _parent
            self.start = self._io.pos() + start_offset  # + (_parent.start if _parent is not None else 0)
            self._root = _root if _root else self
            self._read(start_offset=start_offset)
            self.end = self._io.pos() + start_offset

        def _read(self, start_offset):
            self.version_made_by = self._io.read_u2le()
            self.version_needed_to_extract = self._io.read_u2le()
            self.flags = self._io.read_u2le()
            self.compression_method = KaitaiStream.resolve_enum(Zip.Compression, self._io.read_u2le())
            self._raw_file_mod_time = self._io.read_bytes(4)
            _io__raw_file_mod_time = KaitaiStream(BytesIO(self._raw_file_mod_time))
            self.file_mod_time = dos_datetime.DosDatetime(_io__raw_file_mod_time, self, self._root)
            self.crc32 = self._io.read_u4le()
            self.len_body_compressed = self._io.read_u4le()
            self.len_body_uncompressed = self._io.read_u4le()
            len_filename_io_pos = self._io.pos()
            self.len_file_name = self._io.read_u2le()
            # we might want to check that the filename is not unreasonably long
            len_extra_io_pos = self._io.pos()
            self.len_extra = self._io.read_u2le()
            # len_extra must be a multiple of 4!
            len_comment_io_pos = self._io.pos()
            self.len_comment = self._io.read_u2le()
            self.disk_number_start = self._io.read_u2le()
            self.int_file_attr = self._io.read_u2le()
            self.ext_file_attr = self._io.read_u4le()
            self.ofs_local_header = self._io.read_s4le()
            exception_at = self._io.pos()
            try:
                self._raw_file_name = self._io.read_bytes(self.len_file_name)
            except EOFError as ex:
                allowed_bytes = int(ex.args[0].split("but only ")[1].split(" bytes available")[0])
                raise InvalidDataException("/types/filename/invalid", len_filename_io_pos, expected=allowed_bytes)
            self.file_name = self._raw_file_name.decode(u"UTF-8", "ignore")
            position_of_signature = Zip.find_signtures(self._raw_file_name)
            if len(self._raw_file_name) != self.len_file_name or position_of_signature is not None:
                # filename-length should be:
                # filename_length = filename_length - position_of_signature - len_extra - len_comment (TODO: check if "-1" ???)
                expected_val = self.len_file_name - max(0,
                                                        self.len_file_name - position_of_signature - self.len_extra - self.len_comment)
                raise InvalidDataException("/types/filename/invalid", len_filename_io_pos, expected=expected_val)

            # if length of filename is shorter than expected, we have to guess if the length field is corrupt, or if
            # the filename is corrupt and thus contains illegal bytes...
            exception_at = self._io.pos()
            try:
                self._raw_extra = self._io.read_bytes(self.len_extra)
            except EOFError as ex:
                allowed_bytes = int(ex.args[0].split("but only ")[1].split(" bytes available")[0])
                raise InvalidDataException("/types/extra/invalid", len_extra_io_pos, expected=allowed_bytes)
            position_of_signature = Zip.find_signtures(self._raw_extra)
            if position_of_signature is not None:
                expected_val = self.len_extra - max(0, self.len_extra - position_of_signature - self.len_comment)
                raise InvalidDataException("/types/extra/invalid", len_extra_io_pos, expected=expected_val)
            _io__raw_extra = KaitaiStream(BytesIO(self._raw_extra))
            try:
                self.extra = Zip.Extras(_io__raw_extra, self, self._root, start_offset=start_offset)
            except EOFError as ex:
                allowed_bytes = int(ex.args[0].split("but only ")[1].split(" bytes available")[0])
                raise InvalidDataException("/types/extra/invalid", len_extra_io_pos, expected=allowed_bytes)
            exception_at = self._io.pos()
            try:
                comment = self._io.read_bytes(self.len_comment)
            except EOFError as ex:
                allowed_bytes = int(ex.args[0].split("but only ")[1].split(" bytes available")[0])
                raise InvalidDataException("/types/comment/invalid", len_comment_io_pos, expected=allowed_bytes)
            self.comment = comment.decode(u"UTF-8", "ignore")
            position_of_signature = Zip.find_signtures(comment)
            # set to max unsigned 2 bit value:
            diff_position = None
            if len(comment) != self.len_comment:
                diff_position = Zip.find_first_difference(comment, self.comment.encode("utf-8"))
            if diff_position is not None and position_of_signature is None:
                expected_val = self.len_comment - max(0, self.len_comment - diff_position)
                raise InvalidDataException("/types/comment/invalid", len_comment_io_pos, expected=expected_val)
            elif diff_position is None and position_of_signature is not None:
                expected_val = self.len_comment - max(0, self.len_comment - position_of_signature)
                raise InvalidDataException("/types/comment/invalid", len_comment_io_pos, expected=expected_val)
            elif diff_position is not None and position_of_signature is not None:
                expected_val = self.len_comment - max(0, self.len_comment - min(diff_position, position_of_signature))
                raise InvalidDataException("/types/comment/invalid", len_comment_io_pos, expected=expected_val)

        @property
        def local_header(self):
            if hasattr(self, '_m_local_header'):
                return self._m_local_header

            _pos = self._io.pos()
            self._io.seek(self.ofs_local_header)
            self._m_local_header = Zip.PkSection(self._io, self, self._root, self.start_offset)
            self._io.seek(_pos)
            return getattr(self, '_m_local_header', None)

    class PkSection(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, start_offset=0):
            self._io = _io
            self._parent = _parent
            self.start = self._io.pos() + start_offset  # + (_parent.start if _parent is not None else 0)
            self._root = _root if _root else self
            self._read(start_offset=start_offset)
            self.end = self._io.pos() + start_offset

        def _read(self, start_offset=0):
            self.magic = self._io.read_bytes(2)
            if self.magic != b"\x50\x4B":
                raise kaitaistruct.ValidationNotEqualError(b"\x50\x4B", self.magic, self._io,
                                                           u"/types/pk_section/seq/0")
            self.section_type = self._io.read_u2le()
            _on = self.section_type
            if _on == 513:
                self.body = Zip.CentralDirEntry(self._io, self, self._root, start_offset=start_offset)
            elif _on == 1027:
                self.body = Zip.LocalFile(self._io, self, self._root, start_offset=start_offset)
            elif _on == 1541:
                self.body = Zip.EndOfCentralDir(self._io, self, self._root, start_offset=start_offset)
            elif _on == 2055:
                self.body = Zip.DataDescriptor(self._io, self, self._root, start_offset=start_offset)
            else:
                raise kaitaistruct.ValidationNotAnyOfError(self.section_type, self._io, u"/types/pk_section/seq/1")

    class Extras(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, start_offset=0):
            self._io = _io
            self._parent = _parent
            self.start = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset
            self._root = _root if _root else self
            self._read(start_offset=start_offset)
            self.end = self._io.pos() + (_parent.start if _parent is not None else 0) + start_offset

        def _read(self, start_offset):
            self.entries = []
            i = 0
            while not self._io.is_eof():
                self.entries.append(Zip.ExtraField(self._io, self, self._root, start_offset))
                i += 1

    class LocalFileHeader(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, start_offset=0):
            self._io = _io
            self._parent = _parent
            self.start = self._io.pos() + start_offset  # + (_parent.start if _parent is not None else 0)
            self._root = _root if _root else self
            self._read(start_offset=start_offset)
            self.end = self._io.pos() + start_offset

        def _read(self, start_offset=0):
            self.version = self._io.read_u2le()
            self._raw_flags = self._io.read_bytes(2)
            _io__raw_flags = KaitaiStream(BytesIO(self._raw_flags))
            self.flags = Zip.LocalFileHeader.GpFlags(_io__raw_flags, self, self._root, start_offset)
            self.compression_method = KaitaiStream.resolve_enum(Zip.Compression, self._io.read_u2le())
            self._raw_file_mod_time = self._io.read_bytes(4)
            _io__raw_file_mod_time = KaitaiStream(BytesIO(self._raw_file_mod_time))
            self.file_mod_time = dos_datetime.DosDatetime(_io__raw_file_mod_time, self, self._root)
            self.crc32 = self._io.read_u4le()
            self.len_body_compressed = self._io.read_u4le()
            # TODO we should check if len_body_compress is unreasonalbe (e.g. if there is a valid signature in it)
            self.len_body_uncompressed = self._io.read_u4le()
            len_filename_io_pos = self._io.pos()
            self.len_file_name = self._io.read_u2le()
            len_extra_io_pos = self._io.pos()
            self.len_extra = self._io.read_u2le()
            # len_extra must be a multiple of 4!
            allowed_bytes = None
            exception_at = self._io.pos()
            try:
                self._raw_file_name = self._io.read_bytes(self.len_file_name)
            except EOFError as ex:
                allowed_bytes = int(ex.args[0].split("but only ")[1].split(" bytes available")[0])
                raise InvalidDataException("/types/filename/invalid", len_filename_io_pos, expected=allowed_bytes)
            self.file_name = self._raw_file_name.decode(u"UTF-8", "ignore")
            position_of_signature = Zip.find_signtures(self._raw_file_name)
            diff_position = None
            if len(self._raw_file_name) != len(self.file_name):
                diff_position = Zip.find_first_difference(self._raw_file_name, self.file_name.encode("UTF-8"))
            if diff_position is not None or allowed_bytes is not None or position_of_signature is not None:
                min_error_pos = min([x for x in [diff_position, allowed_bytes, position_of_signature] if x is not None])
                expected_val = self.len_file_name - (self.len_file_name - min_error_pos - self.len_extra)
                raise InvalidDataException("/types/filename/invalid", len_filename_io_pos, expected=expected_val)

            allowed_bytes = None
            try:
                exception_at = self._io.pos()
                raw_extra_start_pos = self._io.pos()
                self._raw_extra = self._io.read_bytes(self.len_extra)
            except EOFError as ex:
                allowed_bytes = int(ex.args[0].split("but only ")[1].split(" bytes available")[0])
                raise InvalidDataException("/types/raw_extra/invalid", len_extra_io_pos, expected=allowed_bytes)
            position_of_signature = Zip.find_signtures(self._raw_extra)
            if allowed_bytes is not None or position_of_signature is not None:
                min_error_pos = min([x for x in [allowed_bytes, position_of_signature] if x is not None])
                expected_val = self.len_file_name - (self.len_file_name - min_error_pos - self.len_extra)
                raise InvalidDataException("/types/raw_extra/invalid", len_filename_io_pos, expected=expected_val)
            _io__raw_extra = KaitaiStream(BytesIO(self._raw_extra))
            self.extra = Zip.Extras(_io__raw_extra, self, self._root, start_offset=start_offset + raw_extra_start_pos)

        class GpFlags(KaitaiStruct):
            """
            .. seealso::
               - 4.4.4 - https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
            
            
            .. seealso::
               Local file headers - https://users.cs.jmu.edu/buchhofp/forensics/formats/pkzip.html
            """

            class DeflateMode(Enum):
                normal = 0
                maximum = 1
                fast = 2
                super_fast = 3

            def __init__(self, _io, _parent=None, _root=None, start_offset=0):
                self._io = _io
                self._parent = _parent
                self.start = self._io.pos() + (_parent.start if _parent is not None else 0)
                self._root = _root if _root else self
                self._read()
                self.end = self._io.pos() + (_parent.start if _parent is not None else 0)

            def _read(self):
                self.file_encrypted = self._io.read_bits_int_le(1) != 0
                self.comp_options_raw = self._io.read_bits_int_le(2)
                self.has_data_descriptor = self._io.read_bits_int_le(1) != 0
                self.reserved_1 = self._io.read_bits_int_le(1) != 0
                self.comp_patched_data = self._io.read_bits_int_le(1) != 0
                self.strong_encrypt = self._io.read_bits_int_le(1) != 0
                self.reserved_2 = self._io.read_bits_int_le(4)
                self.lang_encoding = self._io.read_bits_int_le(1) != 0
                self.reserved_3 = self._io.read_bits_int_le(1) != 0
                self.mask_header_values = self._io.read_bits_int_le(1) != 0
                self.reserved_4 = self._io.read_bits_int_le(2)

            @property
            def deflated_mode(self):
                if hasattr(self, '_m_deflated_mode'):
                    return self._m_deflated_mode

                if ((self._parent.compression_method == Zip.Compression.deflated) or (
                        self._parent.compression_method == Zip.Compression.enhanced_deflated)):
                    self._m_deflated_mode = KaitaiStream.resolve_enum(Zip.LocalFileHeader.GpFlags.DeflateMode,
                                                                      self.comp_options_raw)

                return getattr(self, '_m_deflated_mode', None)

            @property
            def imploded_dict_byte_size(self):
                """8KiB or 4KiB in bytes."""
                if hasattr(self, '_m_imploded_dict_byte_size'):
                    return self._m_imploded_dict_byte_size

                if self._parent.compression_method == Zip.Compression.imploded:
                    self._m_imploded_dict_byte_size = ((8 if (self.comp_options_raw & 1) != 0 else 4) * 1024)

                return getattr(self, '_m_imploded_dict_byte_size', None)

            @property
            def imploded_num_sf_trees(self):
                if hasattr(self, '_m_imploded_num_sf_trees'):
                    return self._m_imploded_num_sf_trees

                if self._parent.compression_method == Zip.Compression.imploded:
                    self._m_imploded_num_sf_trees = (3 if (self.comp_options_raw & 2) != 0 else 2)

                return getattr(self, '_m_imploded_num_sf_trees', None)

            @property
            def lzma_has_eos_marker(self):
                if hasattr(self, '_m_lzma_has_eos_marker'):
                    return self._m_lzma_has_eos_marker

                if self._parent.compression_method == Zip.Compression.lzma:
                    self._m_lzma_has_eos_marker = (self.comp_options_raw & 1) != 0

                return getattr(self, '_m_lzma_has_eos_marker', None)

    class EndOfCentralDir(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None, start_offset=0):
            self._io = _io
            self._parent = _parent
            self.start = self._io.pos() + start_offset  # + (_parent.start if _parent is not None else 0)
            self._root = _root if _root else self
            self._read()
            self.end = self._io.pos() + start_offset

        def _read(self):
            self.disk_of_end_of_central_dir = self._io.read_u2le()
            self.disk_of_central_dir = self._io.read_u2le()
            self.num_central_dir_entries_on_disk = self._io.read_u2le()
            self.num_central_dir_entries_total = self._io.read_u2le()
            self.len_central_dir = self._io.read_u4le()
            self.ofs_central_dir = self._io.read_u4le()
            len_comment_io_pos = self._io.pos()
            self.len_comment = self._io.read_u2le()
            allowed_bytes = None
            try:
                exception_at = self._io.pos()
                raw_extra_start_pos = self._io.pos()
                comment = (self._io.read_bytes(self.len_comment))
                self.comment = comment.decode(u"UTF-8", "ignore")
            except EOFError as ex:
                allowed_bytes = int(ex.args[0].split("but only ")[1].split(" bytes available")[0])
                raise InvalidDataException("/types/comment_len/invalid", len_comment_io_pos, expected=allowed_bytes)
            position_of_signature = Zip.find_signtures(comment)
            if len(comment) != self.len_comment or position_of_signature is not None:
                if position_of_signature is None:
                    position_of_signature = 0
                expected_val = self.len_comment - max(0,
                                                      self.len_comment - position_of_signature)
                raise InvalidDataException("/types/comment/invalid", len_comment_io_pos, expected=expected_val)
