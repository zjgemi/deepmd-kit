import unittest
import numpy as np
import os
import shutil
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.utils.data import DeepmdData

class TestDensityDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test data
        cls.test_dir = os.path.join(os.path.dirname(__file__), "dataset/Al13Cu3")
        os.makedirs(cls.test_dir, exist_ok=True)
    
    def test_density_data_loading(self):
        data = DeepmdData(self.test_dir, density_grid_size=(10, 1, 1))
        data.add("density", 1, atomic=False, must=True, high_prec=False)
        
        # Load the data
        loaded_data = data._load_set(os.path.join(self.test_dir, "set.000"))

        # Check if density data is loaded correctly
        self.assertIn("density", loaded_data)
        self.assertEqual(loaded_data["density"].shape, (1, 10)) # 1 frames, 10x1x1 grid
    
    def test_grid_generation(self):
        data = DeepmdData(self.test_dir, density_grid_size=(10, 1, 1))
        data.add("grid", 3, atomic=False, must=True, high_prec=False)
        
        # Load the data
        loaded_data = data._load_set(os.path.join(self.test_dir, "set.000"))

        # Check if grid data is generated correctly
        self.assertIn("grid", loaded_data)
        self.assertEqual(loaded_data["grid"].shape, (1, 10, 3)) # 1 frames, 10x1x1 grid, 3 dim
    
    def test_density_data_shape(self):
        data = DeepmdData(self.test_dir, density_grid_size=(1, 1, 1))
        data.add("density", 1, atomic=False, must=True, high_prec=False)
        
        # Get a batch of data
        batch_data = data.get_batch(1)

        # Check if density data has the correct shape in the batch
        self.assertIn("density", batch_data)
        self.assertEqual(batch_data["density"].shape, (1, 1))  # 1 frame, 1x1x1 grid

    def test_grid_data_shape(self):
        data = DeepmdData(self.test_dir, density_grid_size=(1, 1, 1))
        data.add("grid", 3, atomic=False, must=True, high_prec=False)
        
        # Get a batch of data
        batch_data = data.get_batch(1)

        # Check if grid data has the correct shape in the batch
        self.assertIn("grid", batch_data)
        self.assertEqual(batch_data["grid"].shape, (1, 1, 3)) # 1 frame, 1x1x1 grid, 3 dim

if __name__ == '__main__':
    unittest.main()