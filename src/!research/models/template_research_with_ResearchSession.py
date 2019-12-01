from src import *
from data.references import DATASETS
import src.config as cfg
import os
import sys
import research_config
from src.utils.model_analysis import ResearchSession

if __name__ == "__main__":
    rs = ResearchSession(research_config, 'models/my_pretrained')
    rs.start_research_session()
