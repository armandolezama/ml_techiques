from Analysis_module import Analysis, Data_module, Graphics

class Report_module:
  def __init__(self):
    self.data_module = Data_module()
    self.analysis_module = Analysis()
    self.graphics_module = Graphics()
    print('hi from report module')