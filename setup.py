import setuptools


setuptools.setup(name='cvsutils',
                 version='0.1.4',
                 description="Unofficial utility scripts for Microsoft Custom Vision Service",
                 packages=setuptools.find_packages(),
                 license='MIT',
                 install_requires=['tqdm', 'Pillow', 'requests', 'tenacity'],
                 url='https://github.com/shonohs/cvsutils',
                 classifiers=[
                     'Intended Audience :: Developers',
                     'License :: OSI Approved :: MIT License'
                 ],
                 entry_points={
                     'console_scripts': [
                         'cvs_create_project=cvsutils.commands.create_project:main',
                         'cvs_download_project=cvsutils.commands.download_project:main',
                         'cvs_evaluate_project=cvsutils.commands.evaluate_project:main',
                         'cvs_export_model=cvsutils.commands.export_model:main',
                         'cvs_get_domains=cvsutils.commands.get_domains:main',
                         'cvs_list_projects=cvsutils.commands.list_projects:main',
                         'cvs_predict_image=cvsutils.commands.predict_image:main',
                         'cvs_remove_iteration=cvsutils.commands.remove_iteration:main',
                         'cvs_train_project=cvsutils.commands.train_project:main',
                         'cvs_validate_dataset=cvsutils.commands.validate_dataset_file:main'
                     ]
                 })
