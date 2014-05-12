import orbit
import unittest
import numpy
import math

class TestEllipticalOrbit(unittest.TestCase):

    def setUp(self):
        self.circularShip = orbit.DynamicBody(numpy.array([7.85,0.0], dtype=float), numpy.array([0.0, 3.2]))
        self.planet = orbit.StaticBody(numpy.array([0.0, 0.0], dtype=float), 80)
        self.circularOrbit = orbit.Orbit(self.circularShip, self.planet)

    def test_circular_orbit_eccentricity(self):
        """Tests a *nearly* circular orbit."""

        # True circular orbits have a 0 eccentricity; this orbit is near circular so we build in a tolerence.
        self.assertTrue(self.circularOrbit.e < .01)

        # Test various options on displayPoints()
        self.assertTrue(len([i for i in self.circularOrbit.displayPoints()]) == 31)
        self.assertTrue(len([i for i in self.circularOrbit.displayPoints(close=False)]) == 30)
        self.assertTrue(len([i for i in self.circularOrbit.displayPoints(points=100, close=False)]) == 100)

        for point in self.circularOrbit.displayPoints():
            radius = numpy.linalg.norm(point)
            self.assertTrue(radius > 7.85)
            self.assertTrue(radius < 7.95)

    def test_circular_orbit_true_anomaly(self):
        """Tests the angular parameters of the orbit."""

        divisions = 10
        timePerDivision = self.circularOrbit.period/divisions
        anglePerDivision = math.pi*2/divisions

        # Loop through angles in the near-circular orbit and ensure they are close to regular intervals.
        for division in xrange(divisions):
            time = timePerDivision*division
            angle = self.circularOrbit.trueAnomaly(time)
            error = angle - (anglePerDivision*division)

            self.assertTrue(error < .01)

    def test_orbit_types(self):
        """Tests to see if the correct subclass is instantiated by the Orbit factory."""

        slowShip = orbit.DynamicBody(numpy.array([-10.5, 0.0], dtype=float) ,numpy.array([0.0, 1.10], dtype=float))
        standardShip = orbit.DynamicBody(numpy.array([-16.64, 2.2219], dtype=float) ,numpy.array([0.4240792, 2.147133], dtype=float))
        fastShip = orbit.DynamicBody(numpy.array([-16.64, 2.2219], dtype=float) ,numpy.array([0.0, 15], dtype=float))
        planet = orbit.StaticBody(numpy.array([0, 0], dtype=float), 80.0)

        shallowOrbit = orbit.Orbit(slowShip, planet)
        ellipticalOrbit = orbit.Orbit(standardShip, planet)
        hyperbolicOrbit = orbit.Orbit(fastShip, planet)

        self.assertTrue(type(shallowOrbit) is orbit.EllipticalOrbit)
        self.assertTrue(type(ellipticalOrbit) is orbit.EllipticalOrbit)
        self.assertTrue(type(hyperbolicOrbit) is orbit.HyperbolicOrbit)


if __name__ == '__main__':
    unittest.main()

