import sys
sys.path.append("../utils")
from report_unit import ReportUnit

class MyFigure(ReportUnit):
    def __init__(self):
        super(MyFigure, self).__init__()

    def run(self):
        str = "hello you"

        self.out_object = str

MyFigure()