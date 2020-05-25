from linearsymplectic import UpperLinearSymplectic
from torch import sigmoid

class UpperNonlinearSymplectic(UpperLinearSymplectic):
    def _matrix_calc_top(self, symmetric_matrix, x_top, x_bottom):
        return x_top + self.h*sigmoid(x_bottom)
    
    def _matrix_calc_bottom(self, symmetric_matrix, x_top, x_bottom):
        return x_bottom

class LowerNonlinearSymplectic(UpperLinearSymplectic):
    def _matrix_calc_top(self, symmetric_matrix, x_top, x_bottom):
        return x_top
   
    def _matrix_calc_bottom(self, symmetric_matrix, x_top, x_bottom):
        return self.h*sigmoid(x_top) + x_bottom