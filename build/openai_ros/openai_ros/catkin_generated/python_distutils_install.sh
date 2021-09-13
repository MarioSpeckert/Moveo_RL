#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/nils/Documents/moveo_ws/src/openai_ros/openai_ros"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/nils/Documents/moveo_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/nils/Documents/moveo_ws/install/lib/python3/dist-packages:/home/nils/Documents/moveo_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/nils/Documents/moveo_ws/build" \
    "/home/nils/anaconda3/bin/python3" \
    "/home/nils/Documents/moveo_ws/src/openai_ros/openai_ros/setup.py" \
     \
    build --build-base "/home/nils/Documents/moveo_ws/build/openai_ros/openai_ros" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/nils/Documents/moveo_ws/install" --install-scripts="/home/nils/Documents/moveo_ws/install/bin"
