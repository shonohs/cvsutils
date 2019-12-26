import setuptools

setuptools.setup(name='cvsutils',
                 version='0.0.0',
                 description="Unofficial utility scripts for Microsoft Custom Vision Service",
                 packages=['cvsutils'],
                 license='MIT',
                 install_requires=['tqdm'],
                 entry_points={
                     'console_scripts': [
                         'cvs_create_project=cvsutils.commands.create_project:main',
                         'cvs_download_project=cvsutils.commands.download_project:main',
                         'cvs_export_model=cvsutils.commands.export_model:main',
                         'cvs_get_domains=cvsutils.commands.get_domains:main',
                         'cvs_train_project=cvsutils.commands.train_project:main',
                         'cvs_predict_image=cvsutils.commands.predict_image:main'
                     ]
                 }
)
