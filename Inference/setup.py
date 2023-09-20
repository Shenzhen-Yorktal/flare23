from setuptools import setup, find_namespace_packages

setup(name='nnunet',
      packages=find_namespace_packages(include=["nnunet", "nnunet.*"]),
      version='1.6.6',
      description='nnU-Net. Framework for out-of-the box biomedical image segmentation.',
      url='https://github.com/MIC-DKFZ/nnUNet',
      author='Division of Medical Image Computing, German Cancer Research Center',
      author_email='f.isensee@dkfz-heidelberg.de',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "scikit-image>=0.14",
            "SimpleITK==2.0",
            "scipy",
            "numpy",
            "connected-components-3d",
            "fastremap"
      ],
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net', 'nnunet']
      )
