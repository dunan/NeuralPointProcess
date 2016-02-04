from setuptools import setup

setup(
    name='nnp_baselines',
    version='0.1',
    py_modules=['event_baseline'],
    install_requires=[
        'Click', 'numpy'
    ],
    entry_points='''
        [console_scripts]
        event_baseline=event_baseline:run
        time_baseline=time_baseline:run
    '''
)

