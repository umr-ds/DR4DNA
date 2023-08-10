import norec4dna.GEPP


class FileSpecificRepair:
    def __init__(self, semi_automatic_solver=None, chunk_tag=None, *args,
                 **kwargs):
        self.semi_automatic_solver = semi_automatic_solver
        self.gepp = self.semi_automatic_solver.decoder.GEPP
        self.use_header_chunk = self.semi_automatic_solver.decoder.use_headerchunk
        self.chunk_tag = chunk_tag
        self.active = False

    def on_load(self, *args, **kwargs):
        self.active = True

    def repair(self, *args, **kwargs):
        raise NotImplementedError("Please use a valid subclass of FileSpecificRepair")

    def get_ui_elements(self):
        return []

    def is_compatible(self, *args, **kwargs):
        raise NotImplementedError("Please use a valid subclass of FileSpecificRepair")

    def update_chunk_tag(self, chunk_tag):
        self.chunk_tag = chunk_tag

    def update_gepp(self, gepp):
        # update gepp.b
        pass

    def update_canvas(self, canvas_data):
        # base for all image-based plugins on canvas updates such as reload image or uploading of a repaired image
        pass