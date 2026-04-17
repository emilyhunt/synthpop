"""
Postprocessing module to improve the estimation of F146 wide-filter extinction
with a polynomial function of A_Ks and narrower filter colors.
"""

__all__ = ["CorrectF146Extinction", ]
__author__ = "M.J. Huston"
__date__ = "2026-04-16"

import pandas
import numpy as np
from ._post_processing import PostProcessing
from .convert_mist_mags import ConvertMistMags

class CorrectF146Extinction(PostProcessing):
    """
    Postprocessing module to apply a correction to F146 extinction
    
    Attributes
    ----------
    
    """

    def __init__(self, model, logger, mag_sys='None', **kwargs):
        super().__init__(model,logger, **kwargs)
        if mag_sys != 'AB':
            raise ValueError('To use CorrectF146Extinction, magnitudes must be in AB. Confirm that you '
                'have converted to AB mags before using this module, then set this module\'s mag_sys ' 
                'kwarg to \'AB\' before running. You can put the ConvertMistMags module before this '
                'one in your post_processing_kwargs to do the conversion.')

    @staticmethod
    def calc_Af146(A_Ks, C1, C2, C3):
        # Set up the parameters according to the best-fit result
        a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, \
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = \
            (3.50497611e+00, -1.16332569e+00,  3.67687634e-01, -6.00648328e-02,
            3.86870393e-03, -7.48509923e-01,  2.49950146e-01,  4.34027329e-03,
           -1.08734489e-01,  4.50723303e-01, -6.29187812e-02, -1.04062883e-01,
            8.05804421e-03,  8.43126134e-03, -1.61020681e-02, -6.25995335e-01,
            1.08449894e-01,  6.78048307e-02, -3.75182929e-01,  2.26084042e-01,
            7.06526545e-03, -3.55577257e-02, -3.24932858e-03,  2.06163475e-03,
           -4.28980453e-03, -4.04255647e-01,  2.68961451e-01,  2.72455710e+00,
            5.51738955e+00,  5.25238577e-02,  7.35156340e-02, -7.02143760e-03,
           -7.20934032e-03,  5.48301329e-04,  1.18792578e-01)
        # Plug into the nightmare polynomial
        A_f146_coeff = (a1 + a2*A_Ks + a3*A_Ks**2 + a4*A_Ks**3 + a5*A_Ks**4 +
           b1*C1 + b2*C1**2 + b3*C1**3 + b4*C1**4 + b5*C1*A_Ks + b6*C1**2*A_Ks + 
           b7*C1*A_Ks**2 + b8*C1**2*A_Ks**2 + b9*C1*A_Ks**3 + b10*C1**3*A_Ks + 
           c1*C2 + c2*C2**2 + c3*C2**3 + c4*C2**4 + c5*C2*A_Ks + c6*C2**2*A_Ks + 
           c7*C2*A_Ks**2 + c8*C2**2*A_Ks**2 +  c9*C2*A_Ks**3 + c10*C2**3*A_Ks + 
           d1*C3 + d2*C3**2 + d3*C3**3 + d4*C3**4 + d5*C3*A_Ks + d6*C3**2*A_Ks + 
           d7*C3*A_Ks**2 + d8*C3**2*A_Ks**2 + d9*C3*A_Ks**3 + d10*C3**3*A_Ks)
        return A_f146_coeff * A_Ks

    def do_post_processing(self, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        Run the process
        """
        if 'W146' in dataframe:
            mag_cols = ["Y106", "J129", "H158", "F184", "W146"]
        elif 'm_roman_f146' in dataframe:
            mag_cols = ['m_roman_f106','m_roman_f129','m_roman_f158','m_roman_f184','m_roman_f146']

        # Get absolute mags if needed
        if self.model.parms.obsmag:
            abs_mags = []
            # Convert needed mags back to absolute
            for f in mag_cols:
                abs_mags.append(dataframe[f].to_numpy() - 5*np.log10(100*dataframe['Dist'].to_numpy()) - \
                            self.model.populations[0].extinction.Alambda_Amap(
                                self.model.parms.eff_wavelengths[f]) * \
                            dataframe[self.model.populations[0].extinction.A_or_E_type].to_numpy())
        else:
            abs_mags = [dataframe[f].to_numpy() for f in mag_cols]

        # Get and convert extinction to A_Ks if needed
        A_Ks = dataframe[self.model.populations[0].extinction.A_or_E_type].to_numpy()
        if self.model.populations[0].extinction.A_or_E_type != 'A_Ks':
            print('converting to A_Ks')
            A_Ks *= self.model.populations[0].extinction.Alambda_Amap(2.152152)
            dataframe.loc[:,'A_Ks'] = A_Ks

        # Calculate extinction in F146 filter from the A_Ks and absolute colors
        A_F146 = self.calc_Af146(A_Ks, abs_mags[0]-abs_mags[1],  abs_mags[1]-abs_mags[2],  abs_mags[2]-abs_mags[3])
        # Add it as a column in the data frame
        dataframe.loc[:,'A_F146'] = A_F146

        # If we are using observed magnitudes, convert F146 back
        if self.model.parms.obsmag:
            dataframe.loc[:,mag_cols[-1]] = abs_mags[-1] + 5*np.log10(100*dataframe['Dist'].to_numpy()) + A_F146

        return dataframe
