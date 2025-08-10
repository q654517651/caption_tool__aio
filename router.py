from views.datasets_view import DatasetsView
from views.dataset_detail_view import DatasetDetailView

class Router:
    def __init__(self, page, dataset_manager, terminal_service):
        self.page = page
        self.dm = dataset_manager
        self.term = terminal_service
        self.current = None  # 当前视图实例

    def to_datasets(self):
        # from views.datasets_view import DatasetsView
        v = DatasetsView(self.page, self.dm, self.term, on_open_dataset=self.to_detail)
        self.current = v
        self.page.controls[0].content = v.build()
        self.page.update()
        v.refresh()

    def to_detail(self, dataset_id):
        # from views.dataset_detail_view import DatasetDetailView
        v = DatasetDetailView(self.page, self.dm, self.term, dataset_id, on_back=self.to_datasets)
        self.current = v
        self.page.controls[0].content = v.build()
        self.page.update()
        v.reload_images()
