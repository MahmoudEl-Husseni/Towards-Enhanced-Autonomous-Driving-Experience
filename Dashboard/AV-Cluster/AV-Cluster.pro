QT += quick core serialport gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp\
            serialmanager.cpp

RESOURCES += qml.qrc \
    qml.qrc \
    qml.qrc \
    qml.qrc \
    qml.qrc

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

# Additional import path used to resolve QML modules just for Qt Quick Designer
QML_DESIGNER_IMPORT_PATH =

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    serialmanager.h

DISTFILES += \
    images/Low-bat.png \
    images/Parking-break.png \
    images/assist-disable.png \
    images/door-open.png \
    images/high-beam.png \
    images/left.png \
    images/low-beam.png \
    images/qit91mty-1 - Copy (2).png \
    images/qit91mty-5 - Copy.png \
    images/right.png \
    images/seat-belt.png \
    images/speed-limit.png \
    images/steering-error.png \
    main.qml
