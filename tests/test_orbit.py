import orbit
import unittest
import numpy

class TestEllipticalOrbit(unittest.TestCase):

    def setUp(self):
        self._slowShip = orbit.DynamicBody(numpy.array([-10.5, 0.0], dtype=float) ,numpy.array([0.0, 1.10], dtype=float))
        self._standardShip = orbit.DynamicBody(numpy.array([-16.64, 2.2219], dtype=float) ,numpy.array([0.4240792, 2.147133], dtype=float))
        self._fastShip = orbit.DynamicBody(numpy.array([-16.64, 2.2219], dtype=float) ,numpy.array([0.0, 15], dtype=float))
        self._planet = orbit.StaticBody(numpy.array([0, 0], dtype=float), 80.0)

        self._shallowOrbit = orbit.Orbit(self._slowShip, self._planet)
        self._ellipticalOrbit = orbit.Orbit(self._standardShip, self._planet)
        self._hyperbolicOrbit = orbit.Orbit(self._fastShip, self._planet)

    def test_body_types(self):
        self.assertTrue(type(self._planet) is orbit.StaticBody)
        self.assertTrue(type(self._standardShip) is orbit.DynamicBody)

    def test_orbit_types(self):
        self.assertTrue(type(self._shallowOrbit) is orbit.EllipticalOrbit)
        self.assertTrue(type(self._ellipticalOrbit) is orbit.EllipticalOrbit)
        self.assertTrue(type(self._hyperbolicOrbit) is orbit.HyperbolicOrbit)


if __name__ == '__main__':
    unittest.main()

