QT += quick core gui serialport multimedia

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    serialmanager.cpp

RESOURCES += \
    qml.qrc

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

# Additional import path used to resolve QML modules just for Qt Quick Designer
QML_DESIGNER_IMPORT_PATH =

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    .gitignore \
    MyDisplayArea.qml \
    animation/musicWave.gif \
    file.mp3 \
    fonts/Exo2-Bold.ttf \
    fonts/Exo2-ExtraBold.ttf \
    fonts/Exo2-Medium.ttf \
    fonts/OpenSans-Bold.ttf \
    fonts/OpenSans-BoldItalic.ttf \
    fonts/OpenSans-ExtraBold.ttf \
    fonts/OpenSans-ExtraBoldItalic.ttf \
    fonts/OpenSans-Italic.ttf \
    fonts/OpenSans-Light.ttf \
    fonts/OpenSans-LightItalic.ttf \
    fonts/OpenSans-Medium.ttf \
    fonts/OpenSans-MediumItalic.ttf \
    fonts/OpenSans-Regular.ttf \
    fonts/OpenSans-SemiBold.ttf \
    fonts/OpenSans-SemiBoldItalic.ttf \
    fonts/fonts.txt \
    images/Adaptive-off.png \
    images/Adaptive-on.png \
    images/Camera-screen.png \
    images/Low-bat.png \
    images/Parking-break.png \
    images/Set 1/01.png \
    images/Set 1/02.png \
    images/Set 1/03.png \
    images/Set 1/04.png \
    images/Set 1/05.png \
    images/Set 1/06.png \
    images/Set 1/07.png \
    images/Set 1/08.png \
    images/Set 1/09.png \
    images/Set 1/10.png \
    images/Set 1/11.png \
    images/Set 1/12.png \
    images/Set 1/13.png \
    images/Set 1/14.png \
    images/Set 1/15.png \
    images/Set 1/16.png \
    images/Set 1/17.png \
    images/Set 1/18.png \
    images/Set 1/19.png \
    images/Set 1/20.png \
    images/Set 1/21.png \
    images/Set 1/22.png \
    images/Set 1/23.png \
    images/Set 1/24.png \
    images/Set 1/25.png \
    images/Set 1/26.png \
    images/Set 1/27.png \
    images/Set 1/28.png \
    images/Set 1/29.png \
    images/Set 1/30.png \
    images/Set 1/31.png \
    images/Set 1/32.png \
    images/Set 1/33.png \
    images/Set 1/34.png \
    images/Set 1/35.png \
    images/Set 1/36.png \
    images/Set 1/37.png \
    images/Set 1/38.png \
    images/Set 1/39.png \
    images/Set 1/40.png \
    images/assist-disable.png \
    images/back-button.png \
    images/base_1.png \
    images/bluetooth.png \
    images/calendar.png \
    images/call.png \
    images/clock.png \
    images/dark-mode.png \
    images/door-open.png \
    images/google-maps.png \
    images/hand.png \
    images/high-beam.png \
    images/left.png \
    images/light-mode.png \
    images/logo.png \
    images/love-birds.png \
    images/love.png \
    images/low-beam.png \
    images/map.png \
    images/music.png \
    images/navigation.png \
    images/next-button.png \
    images/nightMode.png \
    images/number-2.png \
    images/number-3.png \
    images/number-5.png \
    images/number-8.png \
    images/number-9.png \
    images/number-four.png \
    images/number-one.png \
    images/pause.png \
    images/play.png \
    images/right.png \
    images/seat-belt.png \
    images/settings.png \
    images/seven.png \
    images/signal-status.png \
    images/signal.png \
    images/six.png \
    images/sound-waves.png \
    images/speed-limit.png \
    images/steering-error.png \
    images/telephone.png \
    images/thermostat.png \
    images/weather.png \
    images/zero.png \
    main.qml \
    main2.qml


HEADERS += \
    serialmanager.h
