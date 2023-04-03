pip install scikit-build
pip install -r requirements.txt  # Install the requirements
cd third_party/progressive-x/graph-cut-ransac; mkdir build; cd build; cmake ..; make -j8; cd ../../../..  # Install the C++ library Graph Cut RANSAC
cd third_party/progressive-x; mkdir build; cd build; cmake ..; make -j8; cd ../../..  # Install the C++ library Progressive-X
pip install -e third_party/progressive-x  # Install the Python bindings of Progressive-X for VP estimation
pip install -e line_refinement  # Install the Python bindings to optimize lines wrt a distance/angle field
pip install -e third_party/homography_est  # Install the code for homography estimation from lines
pip install -e .  # Install DeepLSD
