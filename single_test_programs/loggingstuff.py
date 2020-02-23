import logging
import numpy

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

a = numpy.linspace(0, 1000, 100000)
logging.info('Assigned a linear space to a variable.')

b = numpy.tan(numpy.sqrt(a))
logging.info('Transformation done.')

logging.debug('everyone is brave and stuff')


print('b.mean:', b.mean())
logging.info('Program ended.')