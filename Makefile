# Run pytest and coverage report
pytest:
	./scripts/run_tests.sh

# Type check
type:
	pytype

# Build the doc
doc:
	cd docs && make html

# Check the spelling in the doc
spelling:
	cd docs && make spelling

# Clean the doc build folder
clean:
	cd docs && make clean

# Build docker images
# If you do export RELEASE=True, it will also push them
docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

# PyPi package release
release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload dist/*

# Test PyPi package release
test-release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
