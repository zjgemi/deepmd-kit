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
        cls.test_dir = "/root/dpmdata/Al13Cu3"
        os.makedirs(cls.test_dir, exist_ok=True)
    
    def test_density_data_loading(self):
        data = DeepmdData(self.test_dir, density_grid_size=(10, 10, 10))
        data.add("density", 1, atomic=False, must=True, high_prec=False)
        
        # Load the data
        print(os.path.join(self.test_dir, "set.000"))
        loaded_data = data._load_set(os.path.join(self.test_dir, "set.000"))

        # Check if density data is loaded correctly
        self.assertIn("density", loaded_data)
        self.assertEqual(loaded_data["density"].shape, (100, 1000))
    
    def test_grid_generation(self):
        data = DeepmdData(self.test_dir, density_grid_size=(10, 10, 10))
        data.add("grid", 3, atomic=False, must=True, high_prec=False)
        
        # Load the data
        loaded_data = data._load_set(os.path.join(self.test_dir, "set.000"))

        # Check if grid data is generated correctly
        self.assertIn("grid", loaded_data)
        self.assertEqual(loaded_data["grid"].shape, (100, 1000, 3))
    
    def test_density_data_shape(self):
        data = DeepmdData(self.test_dir, density_grid_size=(10, 10, 10))
        data.add("density", 1, atomic=False, must=True, high_prec=False)
        
        # Get a batch of data
        batch_data = data.get_batch(1)

        # Check if density data has the correct shape in the batch
        self.assertIn("density", batch_data)
        self.assertEqual(batch_data["density"].shape, (100, 1000))  # 100 frame, 10x10x10 grid

    def test_grid_data_shape(self):
        data = DeepmdData(self.test_dir, density_grid_size=(10, 10, 10))
        data.add("grid", 3, atomic=False, must=True, high_prec=False)
        
        # Get a batch of data
        batch_data = data.get_batch(1)

        # Check if grid data has the correct shape in the batch
        self.assertIn("grid", batch_data)
        self.assertEqual(batch_data["grid"].shape, (100, 1000, 3))

if __name__ == '__main__':
    unittest.main()