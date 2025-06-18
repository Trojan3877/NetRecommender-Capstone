# tests/test_collab.py

import unittest
from assistant.collaborative import CollaborativeFiltering

class TestCollaborativeFiltering(unittest.TestCase):
    def setUp(self):
        self.model = CollaborativeFiltering()
        self.user_id = 1

    def test_predict_user_known(self):
        recommendations = self.model.predict(self.user_id, top_n=5)
        self.assertEqual(len(recommendations), 5)
        self.assertTrue(all(isinstance(r, int) for r in recommendations))

    def test_invalid_user(self):
        with self.assertRaises(ValueError):
            self.model.predict(-1)

if __name__ == '__main__':
    unittest.main()
