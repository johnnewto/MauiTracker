from setuptools import setup

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")
    print(install_requires)

setup(
    name='MauiTracker',
    version='v0.0.1',
    packages=['utils', 'motrackers', 'motrackers.utils'],
    url='https://github.com/johnnewto/MauiTracker',
    license='',
    author='john',
    author_email='',
    description='Maui63 CameraTracking',
    install_requires=install_requires,
)
