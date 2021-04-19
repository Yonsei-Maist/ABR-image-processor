from setuptools import setup, find_packages

setup(
    name             = 'abr-image-processor',
    version          = '1.2',
    description      = 'image processor module for ABR',
    author           = 'Chanwoo Gwon',
    author_email     = 'arknell@yonsei.ac.kr',
    url              = 'https://github.com/Yonsei-Maist/ABR-image-processor.git',
    install_requires = [
        "numpy==1.19.5"
    ],
    packages         = find_packages(exclude = ['docs', 'tests*']),
    keywords         = ['opencv', 'abr'],
    python_requires  = '>=3',
    zip_safe=False,
    classifiers      = [
        'Programming Language :: Python :: 3.7'
    ]
)