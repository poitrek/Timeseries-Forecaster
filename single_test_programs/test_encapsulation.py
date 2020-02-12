class Car:
    def __init__(self):
        # Public attribute
        self.plate = 'SK83021'
        # Protected
        self._owner = 'James'
        # Private
        self.__cylinders = 16

    def __str__(self):
        return 'Car no. {} owned by {} has a {}-cylinder engine'.format(self.plate, self._owner, self.__cylinders)

    def get_cylinders(self):
        return self.__cylinders


car1 = Car()
print(car1)
print(car1.plate)
print(car1._owner)
print(car1.get_cylinders())
# Attribute Exception
print(car1.__cylinders)
