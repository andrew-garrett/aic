from setuptools import find_packages, setup

package_name = "aic_lewm_policies"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AIC Team",
    maintainer_email="opensource@intrinsic.ai",
    description="LeWM-focused policy nodes for demonstration collection and training data generation.",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
)
