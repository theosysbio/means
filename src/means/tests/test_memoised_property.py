from __future__ import absolute_import, print_function

import unittest

from means.util.memoisation import memoised_property, MemoisableObject


class A(MemoisableObject):
    def __init__(self, count):
        self.count = count
    @memoised_property
    def f(self):
        self.count += 1
        return ['x', 'y'][self.count]

class B(MemoisableObject):

    def __init__(self, count):
        self.count = count

    @memoised_property
    def f(self):
        self.count += 1
        return ['foobar', 'barfoo'][self.count]



class TestMemoisedProperty(unittest.TestCase):

    def test_memoisation_works(self):
        """
        Given a class that returns a different value each time the property f is called,
        the memoisation should make the returned value to be the same each time
        :return:
        """
        a = A(-1)

        # This should be 'x'
        self.assertEquals(a.f, 'x')
        # This should be 'y' if memoisation did not work
        self.assertEquals(a.f, 'x')

    def test_memoisation_does_not_influence_results_for_function_with_same_name_in_different_class(self):
        """
        Given that two different functions share the same function name, memoisation of these functions
        should not be influenced by one another.
        :return:
        """

        a = A(-1)
        b = B(-1)
        # Memoise something for A
        self.assertEquals(a.f, 'x')

        # Check that c returns correct value
        self.assertEquals(b.f, 'foobar')

        # Check that memoisation still works for both of them
        self.assertEquals(a.f, 'x')
        self.assertEquals(b.f, 'foobar')

    def test_memoisation_does_not_persist_for_different_instances_of_class(self):
        """
        Given two instances of the same class, the memoisation should not persist between the two
        """

        a = A(-1)
        b = A(0)

        # Memoise
        self.assertEquals(a.f, 'x')

        # This should return 1 as counter=0 initially
        self.assertEquals(b.f, 'y')

        # Check if memoisation still works
        self.assertEquals(a.f, 'x')
        self.assertEquals(b.f, 'y')
