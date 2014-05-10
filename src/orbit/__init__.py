# Copyright (c) 2014 Michael Kessler
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy
import math

# Support functions
def normalize(vector):
    # Returns a normalized copy of the input vector.
    return vector / numpy.linalg.norm(vector)

def angleBetween(vec1, vec2):
    # Returns the angle between two vectors.
    nvec1 = normalize(vec1)
    nvec2 = normalize(vec2)
    return numpy.arccos(numpy.dot(nvec1, nvec2))


class StaticBody(object):
    """Represents a fixed position gravitational source."""

    def __init__(self, position, mass):
        """
        (numpy.array) Position of the center of mass on a 2d plane.

        (float) Mass of the the body.

        """

        self._position = position
        self._mass = mass


    @property
    def mass(self):
        return self._mass

    @property
    def position(self):
        return self._position


class DynamicBody(object):
    """Represents an object in freeflight around a gravitational source."""

    def __init__(self, position, velocity):
        """
        (numpy.array) Initial position of the orbiting body.

        (numpy.array) Initial velicty of the orbiting body.

        """

        self._position = position
        self._velocity = velocity

    @property
    def position(self):
        return self._position

    @property
    def velocity(self):
        return self._velocity


class Orbit(object):
    """Represents the path of an orbiting object in a 2-body system.

    """

    # Standard gravitational constant.  For real scaling, see the
    # following wikipedia article:
    # http://en.wikipedia.org/wiki/Standard_gravitational_parameter
    g = 1.0;

    def __new__(cls, orbiter, target):
        """
        (DynamicBody) The object to be considered as the orbiting body.

        (StaticBody) The gravity source for the orbiting body to orbit.

        """

        # This is both the base class and factory for generating orbits.
        # Having this the factory makes the interface easy to use while
        # still keeping the code maintainable.  To determine what kind
        # of orbit we have, first we need to find the burnout angle.
        # Once we have the burnout angle, we have everything we need to
        # compute the eccentricity, which is what determines what kind
        # of orbit we have.  To do this, we will create the most common
        # orbit type first (elliptical), compute the two terms, and if
        # the eccentricity is greater than 1.0 we will instead construct
        # a hyperbolic orbit.

        orbit = EllipticalOrbit(orbiter, target)

        # If the eccentricity is greater than 1.0, we actually have a
        # hyperbolic orbit, construct and return that instead.
        if orbit.e > 1.0:
            orbit = HyperbolicOrbit(orbiter, target)

        print orbit.apoapsis, orbit.periapsis
            
        return orbit


    def __init__(self, orbiter, target):

        # Direct assignments
        self._orbiter = orbiter
        self._target = target

        # Implicit orbital terms
        self._burnoutAngle = None
        self._e = None
        self._ngtoorb = None
        self._gm = None
        self._periapsis = None
        self._apoapsis = None

    def _computeApsi(self):
        r = self.r
        v = numpy.linalg.norm(self.orbiter.velocity)
        M = self.target.mass
        nvel = normalize(self.orbiter.velocity)

        angle = self.burnoutAngle
        GM = self.gm

        C = 2.0 * GM / (r*(v**2))
        negC = C*-1.0
        omC = 1.0-C
        C2 = C**2

        # Perform the quadratic equation to determine our apsi
        quadSecond = math.sqrt(
            C2 - (4.0 * omC * -1.0 * (math.sin(angle)**2))
        )

        quadDivisor = 2.0 * omC

        Rp1 = (negC + quadSecond) / quadDivisor
        Rp2 = (negC - quadSecond) / quadDivisor

        # This happens when we have an eccentricity > 1.0
        if (Rp1 > Rp2):
            self._apoapsis = Rp1*r
            self._periapsis = Rp2*r
        else:
            self._apoapsis = Rp2*r
            self._periapsis = Rp1*r

        if (Rp1 < 0.0):
            self._periapsis = Rp2*r
        elif (Rp2 < 0.0):
            self._periapsis = Rp1*r



    @property
    def apoapsis(self):
        if self._apoapsis is None:
            self._computeApsi()
        return self._apoapsis


    @property
    def burnoutAngle(self):
        """The angle between the gravitational source and where the orbit began."""
        if self._burnoutAngle is None:
            nvel = normalize(self.orbiter.velocity)
            angle = (angleBetween(self.ngtoorb, nvel)/180.0)*math.pi
            cross = numpy.cross(self.ngtoorb, nvel)

            if (numpy.dot(numpy.array([0.0, -1.0], dtype=float), cross)[0] < 0.0):
                angle *= -1.0;

        return angle

    @property
    def e(self):
        """The orbit's eccentricity term."""
        if self._e is None:
            r = self.r
            v = numpy.linalg.norm(self.orbiter.velocity)
            self._e = numpy.sqrt(
                ((r*(v**2) / self.gm - 1.0)**2) *
                (math.sin(self.burnoutAngle))**2 +
                (math.cos(self.burnoutAngle))**2
            )

        return self._e

    @property
    def gm(self):
        if self._gm is None:
            self._gm = self.g*self.target.mass
        return self._gm

    @property
    def ngtoorb(self):
        """The normalized gravitational source to orbiting start position."""
        if self._ngtoorb is None:
            self._ngtoorb = normalize(self.target.position-self.orbiter.position)

        return self._ngtoorb

    @property
    def orbiter(self):
        """The object considered to be the orbiting body."""
        return self._orbiter

    @property
    def periapsis(self):
        if self._periapsis is None:
            self._computeApsi()
        return self._periapsis

    @property
    def r(self):
        return numpy.linalg.norm(self.orbiter.position - self.target.position)

    @property
    def target(self):
        """The object considered to be the gravity source."""
        return self._target


class EllipticalOrbit(Orbit):
    def __new__(cls, orbiter, target):
        instance = object.__new__(cls, orbiter, target)
        return instance

class HyperbolicOrbit(Orbit):
    def __new__(cls, orbiter, target):
        instance = object.__new__(cls, orbiter, target)
        return instance

