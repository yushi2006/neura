from setuptools import setup


setup(
    name="neura",
    version="0.1.0",
    description="A deep learning library with Python and CUDA",
    packages=["neura", "neura.nn", "neura.cuda", "neura.utils", "neura.api"],
    package_dir={"neura": "src"},
    install_requires=["numpy", "pytest"],
    author="Yusuf Shihata",
    author_email="yusufshihata2006@gmail.com",
    license="MIT",
)
