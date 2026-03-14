
from core.queue_manager import QueueManager
from core.config import ConfigManager
from core.advisor import AnalysisAdvisor
from core.workspace import WorkspaceManager

# Global Singletons for GUI
queue_manager = QueueManager()
config = ConfigManager()
advisor = AnalysisAdvisor()
workspace = WorkspaceManager()

# Global State
was_working = False