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

"""An orbit computation library for 2-body keplerian orbits."""

import numpy
import math

# Note to Programmer:
# Many of the properties on the orbit classes are single-letter properties and
# in a few cases are uppercased.  These parameters most closely represent the
# orbital mechanics common names and should generally match text books and
# other reference.  Below are a few good online references.
#
# http://www.braeunig.us/space/
# http://en.wikipedia.org/wiki/Orbital_mechanics

def normalize(vector):
    """Returns a normalized copy of the input vector."""
    return vector / numpy.linalg.norm(vector)

def angleBetween(vec1, vec2):
    """Returns the angle between two vectors.

    Args:
        vec1 (numpy.array): First input vector
        vec2 (numpy.array): Second input vector

    Returns:
        float

    """
    nvec1 = normalize(vec1)
    nvec2 = normalize(vec2)
    return numpy.arccos(numpy.dot(nvec1, nvec2))

def rotate(vector, angle):
    """Rotates a vector by a given angle.

    Args:
        vector (numpy.array): The input 2d vector as a numpy.array.
        angle (float): The angle to rotate the given vector in radians.

    Returns:
        numpy.array: A 2d array representing the rotated input.

    """

    x = vector[0] * math.cos(angle) - vector[1] * math.sin(angle)
    y = vector[0] * math.sin(angle) + vector[1] * math.cos(angle)
    return numpy.array([x, y], dtype=float)


class StaticBody(object):
    """Represents a fixed position gravitational source."""

    def __init__(self, position, mass):
        """
        (numpy.array) Position of the center of mass on a 2d plane.

        (float) Mass of the the body.

        """

        self._position = position
        self._mass = mass

    def __repr__(self):
        return 'StaticBody([{p[0]},{p[1]}], {m})'.format(
            p=self.position,
            m=self.mass
        )

    @property
    def mass(self):
        """The mass of the gravitational source."""
        return self._mass

    @property
    def position(self):
        """The fixed position of the gravitational source."""
        return self._position


class DynamicBody(object):
    """Represents an object in freeflight around a gravitational source."""

    def __init__(self, position, velocity):
        """A moving body in orbit.

        Args:
            position (numpy.array): Initial position of the orbiting body.
            velocity (numpy.array): Initial velicty of the orbiting body.

        """

        self._position = position
        self._velocity = velocity

    def __repr__(self):
        return 'StaticBody([{p[0]},{p[1]}], [{v[0]},{v[1]}])'.format(
            p=self.position,
            m=self.velocity
        )

    @property
    def position(self):
        """The initial position of the moving body."""
        return self._position

    @property
    def velocity(self):
        """The initial velocity of the moving body."""
        return self._velocity


class Orbit(object):
    """Represents the path of an orbiting object in a 2-body system."""

    # Standard gravitational constant.
    # http://en.wikipedia.org/wiki/Standard_gravitational_parameter
    g = 1.0
    # For real scaling set g to 6.67*10^-11, but for the purposes
    # of floating point accuracy and game scales, we use 1.0

    clockwise = 1
    counterclockwise = -1

    def __new__(cls, orbiter, target):
        # This is both the base class and factory for generating orbits.
        # Having this the factory makes the interface easy to use while
        # still keeping the code maintainable.  To determine what kind
        # of orbit we have, first we need to find the burnout angle.
        # Once we have the burnout angle, we have everything we need to
        # compute the eccentricity, which is what determines what kind
        # of orbit we have.  To do this, we will create the most common
        # orbit type first (elliptical), compute the eccentricity, and if
        # the eccentricity is greater than 1.0 we will instead construct
        # a hyperbolic orbit.

        orbit = EllipticalOrbit(orbiter, target)

        # If the eccentricity is greater than 1.0, we actually have a
        # hyperbolic orbit, construct and return that instead.
        if orbit.e > 1.0:
            orbit = HyperbolicOrbit(orbiter, target)

        return orbit


    def __init__(self, orbiter, target):
        """Representation of an orbital path.

        Args:
            orbiter (DynamicBody): The orbiting body.
            target (StaticBody): The gravity source.

        """

        # Direct assignments
        self._orbiter = orbiter
        self._target = target

        # Implicit orbital terms
        self._burnoutAngle = None # Angle of the initial point of orbit
        self._a = None # Semi-major axis
        self._b = None # Semi-minor axis
        self._e = None # Eccentricity
        self._f = None # Distance from center to focci.
        self._ngtoorb = None # Normalized gravity source to orbiting start
        self._gm = None # Standard gravatational parameter (constant times mass)
        self._periapsis = None # Closest distance in orbit
        self._apoapsis = None # Furthest distance in orbit
        self._launchAngle = None # Angle between initial point and periapsis
        self._direction = None # Direction of orbit

    def _computeApsi(self):
        """Compute the periapsis and apoapsis and store the results."""
        r = self.r
        v = numpy.linalg.norm(self.orbiter.velocity)

        angle = self.burnoutAngle

        # A few simplifications to reduce complexity in the full equation.
        C = 2.0 * self.gm / (r*(v**2))
        negC = C*-1.0 # Negative C
        omC = 1.0-C # One Minus C
        C2 = C**2 # C Squared

        # Perform the quadratic equation to determine our apsi
        quadSecond = math.sqrt(
            C2 - (4.0 * omC * -1.0 * (math.sin(angle)**2))
        )

        quadDivisor = 2.0 * omC

        # The quadratic equation creates two solutions
        Rp1 = (negC + quadSecond) / quadDivisor
        Rp2 = (negC - quadSecond) / quadDivisor

        # This happens when we have an eccentricity > 1.0
        if Rp1 > Rp2:
            self._apoapsis = Rp1*r
            self._periapsis = Rp2*r
        else:
            self._apoapsis = Rp2*r
            self._periapsis = Rp1*r

        # In the hyperbolic case, make sure we choose the proper periapsis,
        # one solution will be negative.
        if Rp1 < 0.0:
            self._periapsis = Rp2*r
        elif Rp2 < 0.0:
            self._periapsis = Rp1*r

    @property
    def a(self):
        """Semi-major axis."""
        if self._a is None:
            vmag = numpy.linalg.norm(self.orbiter.velocity)
            self._a = 1.0 / (2.0 / self.r - (vmag**2) / self.gm)
        return self._a

    @property
    def apoapsis(self):
        """Furthest distance in orbit."""
        if self._apoapsis is None:
            # Apoapsis and Periapsis are computed together,
            # the first one called will cache both.
            self._computeApsi()
        return self._apoapsis

    @property
    def b(self):
        """Semi-minor axis."""
        if self._b is None:
            self._b = self._a * math.sqrt(1.0-self.e)
        return self._b

    @property
    def burnoutAngle(self):
        """The angle between the gravity source and where the orbit began."""
        if self._burnoutAngle is None:
            nvel = normalize(self.orbiter.velocity)
            angle = angleBetween(self.ngtoorb, nvel)
            cross = numpy.cross(self.ngtoorb, nvel)

            if numpy.dot(numpy.array([0.0, -1.0], dtype=float), cross)[0] < 0.0:
                angle *= -1.0

        return angle

    @property
    def direction(self):
        """The direction of orbit from the 'top' vantage point."""
        if self._direction is None:
            rightanglevec = rotate(self.ngtoorb, math.pi/2.0)
            if numpy.dot(rightanglevec, normalize(self.orbiter.velocity)) > 0.0:
                self._direction = self.counterclockwise
            else:
                self._direction = self.clockwise
        return self._direction

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
    def f(self):
        """The distance from the center to either focci."""
        if self._f is None:
            self._f = self.e*self.a

        return self._f

    @property
    def gm(self):
        """The standard orbital parameter multiplied by mass of the target."""
        if self._gm is None:
            self._gm = self.g*self.target.mass

        return self._gm

    @property
    def launchAngle(self):
        """The angle between the orbiting object's velocity and the orbital path."""
        if self._launchAngle is None:
            vmag = numpy.linalg.norm(self.orbiter.velocity)
            rvmagsqr = (self.r * (vmag**2) / self.gm)
            sincos = math.sin(self.burnoutAngle) * math.cos(self.burnoutAngle)
            denom = (rvmagsqr * (math.sin(self.burnoutAngle)**2)-1)
            ltp = (rvmagsqr * sincos) / denom
            launchToPeri = math.atan(ltp)

            if denom < 0.0:
                launchToPeri += math.pi

            self._launchAngle = launchToPeri

        return self._launchAngle

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
        """Closest distance of the entire orbital period."""
        if self._periapsis is None:
            self._computeApsi()
        return self._periapsis

    @property
    def r(self):
        """Radius of orbit at the initial position angle."""
        return numpy.linalg.norm(self.orbiter.position - self.target.position)

    @property
    def target(self):
        """The object considered to be the gravity source."""
        return self._target

    def displayPoints(self, points=0):
        """Unimplemented method to be implemented by subclasses.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError("displayPoints() must not be run from a generic orbit.")


class EllipticalOrbit(Orbit):
    """An EllipticalOrbit is the representation of an orbital path of an orbit which
    is captured by the gravity source an will never exit.  The resulting path will
    form the shape of an ellipse.

    """

    def __new__(cls, orbiter, target):
        instance = object.__new__(cls, orbiter, target)
        return instance

    def displayPoints(self, points=30):
        """Generator to create points to display the orbit.

        points (int) Number of points to be used to represent the orbit.

        """

        # The angle per point to sweep based on points
        radPerPoint = (math.pi*2.0) / float(points)

        firstPoint = None
        for i in xrange(0, points):
            theta = i*radPerPoint
            r = self.a*((1 - (self.e**2)) / (1+self.e*math.cos(theta)))
            angle = self.direction*(theta+self.launchAngle+math.pi)
            point = rotate(self.ngtoorb, angle)*r

            if firstPoint is None:
                firstPoint = point

            # Generate one point at at time so drawing routines can iterate.
            yield point

        # Finally end with the starting point to close the loop.
        yield firstPoint

class HyperbolicOrbit(Orbit):
    """A HyperbolicOrbit is the representation of the orbital path of an orbit which
    is not fully captured by the gravitational source.  They have the shape of a
    hyperbola and in a simplified n-body system is considered to acheive an escape
    orbit.

    """

    def __new__(cls, orbiter, target):
        instance = object.__new__(cls, orbiter, target)
        return instance

    def __init__(self, orbiter, target):
        """A hyperbolic/escape tragectory orbit."""
        super(HyperbolicOrbit, self).__init__(orbiter, target)
        self._asymptote = None # The angle from periapsis of which the orbit approaches.
        self._semilatusRectum = None # Hyperbolic term relating to asymptote angles.

    @property
    def asymptote(self):
        """The angle at which the hyperbolic asymptote relative to the semi-latus rectum."""

        if self._asymptote is None:
            self._asymptote = math.acos(-(1.0/self.e))
        return self._asymptote

    def displayPoints(self, points=30):
        """Generator to create points to display the orbit.

        Args:
            points (int): Number of points to be used to represent the orbit.

        Returns:
            generator: Returns a generator returning elements as [x,y] in a numpy.array

        """

        start = -(self.asymptote-.1)
        end = self.asymptote-.1
        radPerPoint = (end-start)/points

        for i in xrange(0, points):
            theta = (i*radPerPoint)+start
            r = self.a*((1 - (self.e**2)) / (1+self.e*math.cos(theta)))
            angle = self.direction*(theta+self.launchAngle+math.pi)
            point = rotate(self.ngtoorb, angle)*r

            # Generate one point at at time so drawing routines can iterate.
            yield point

    @property
    def semilatusRectum(self):
        """The hyperbolic term from which the asymptotes cross.
        Described as a distance from the focus source through the periapsis.
        See: http://mathworld.wolfram.com/SemilatusRectum.html

        """

        if self._semilatusRectum is None:
            self._semilatusRectum = self.a*(1.0-(self.e**2))

        return self._semilatusRectum

