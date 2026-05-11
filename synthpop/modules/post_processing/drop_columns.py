"""
Post-processing module to drop unnecessary columns.
"""

__all__ = ["DropColumns", ]
__author__ = "M.J. Huston"
__date__ = "2026-03-23"

import pandas
import numpy as np
from ._post_processing import PostProcessing

class DropColumns(PostProcessing):
    """
    Post-processing module to rename columns
    
    Attributes
    ----------
    old_names : list
        list of columns to rename
    new_names : list
        list of new names in same order as old_names
    """

    def __init__(self, model, logger, column_names=None, **kwargs):
        super().__init__(model,logger, **kwargs)
        self.column_names = column_names

    def do_post_processing(self, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        Make the column name changes.
        """
        
        dataframe.drop(columns=self.column_names, inplace=True)
        return dataframe
