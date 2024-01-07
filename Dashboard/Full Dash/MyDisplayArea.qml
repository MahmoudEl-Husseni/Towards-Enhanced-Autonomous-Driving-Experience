import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Extras 1.4
import Qt.labs.calendar 1.0
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import QtPositioning 5.12
import QtLocation 5.12
import QtMultimedia 5.12
import QtQuick.Controls.Styles 1.4
Rectangle {
    id: displayArea
    width: 1236
    height: 698
    visible: true
    color: "#2E2F30"
    state: ""
    focus: true



    Rectangle {
        id: progressRec
        x: 331
        y: 547
        width: 600
        height: 20
        visible: false
        color: "#ffffff"

        Rectangle {
            id: audioProgress
            width: 0
            height: parent.height
            visible: false
             color: "#141bb1"
        }
    }







    AnimatedImage {
        id: animatedGif
        x: 367
        y: 174
        width: 500
        height: 500
        source: "animation/musicWave.gif"
        anchors.verticalCenterOffset: -91
        anchors.horizontalCenterOffset: 13
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.fill: displayArea.PreserveAspectFit
        visible: false
        anchors.verticalCenter: parent.verticalCenter
        playing: false

    }




    Image {
        id: backbutton
        x: 321
        y: 565
        height: 100
        visible: false
        source: "images/back-button.png"
        transformOrigin: Item.Bottom
        fillMode: Image.PreserveAspectFit
        MouseArea {
            id: mouseAreaBack
            anchors.fill: parent
            transformOrigin: Item.Bottom
            onClicked: {  mediaPlayer.stop()
                          mediaPlayer.play()}
        }
    }

    Image {
        id: nextbutton
        x: 600
        y: 563
        height: 100
        visible: false
        source: "images/next-button.png"
        transformOrigin: Item.Bottom
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: pause
        x: 921
        y: 563
        height: 100
        visible: false
        source: "images/pause.png"
        transformOrigin: Item.Bottom
        fillMode: Image.PreserveAspectFit
        MouseArea {
            id: mouseAreaPause
            anchors.fill: parent
            transformOrigin: Item.Bottom
            onClicked: {
                mediaPlayer.pause();
                animatedGif.playing = false;
            }
        }
    }

    Image {
        id: play
        x: 382
        y: 563
        height: 100
        opacity: 1
        visible: false
        source: "images/play.png"
        transformOrigin: Item.Bottom
        fillMode: Image.PreserveAspectFit
        MouseArea {
            id: mouseAreaPlay
            anchors.fill: parent
            transformOrigin: Item.Bottom
            onClicked: {
                mediaPlayer.play();
                animatedGif.playing = true;
            }
        }


    }

    Image {
        id: map
        visible: false
        anchors.fill: parent
        source: "images/map.png"
        fillMode: Image.PreserveAspectFit
    }

    Text {
        id: text1
        x: 110
        y: 155
        visible: false
        text: qsTr("Text")
        font.pixelSize: 12

        Image {
            id: clock
            x: 113
            y: -42
            visible: false
            source: "images/clock.png"
            fillMode: Image.PreserveAspectFit
        }
    }

    Text {
        id: text2
        x: 150
        y: 257
        visible: false
        text: qsTr("Text")
        font.pixelSize: 12
    }

    TextInput {
        id: textInput
        x: 449
        y: 164
        width: 80
        height: 20
        visible: false
        text: qsTr("Text Input")
        font.pixelSize: 12
    }

    Image {
        id: calendar
        x: 249
        y: 234
        visible: false
        source: "images/calendar.png"
        fillMode: Image.PreserveAspectFit
    }

    TextInput {
        id: textInput1
        x: 513
        y: 269
        width: 80
        height: 20
        visible: false
        text: qsTr("Text Input")
        font.pixelSize: 12
    }

    Text {
        id: text3
        x: 448
        y: 48
        visible: false
        text: qsTr("Text")
        font.pixelSize: 12
        anchors.horizontalCenter: parent.horizontalCenter
    }

    Image {
        id: numberone
        x: 110
        y: 29
        visible: false
        source: "images/number-one.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: number2
        x: 431
        y: 60
        visible: false
        source: "images/number-2.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: number3
        x: 150
        y: 113
        visible: false
        source: "images/number-3.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: numberfour
        x: 154
        y: 139
        visible: false
        source: "images/number-four.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: number5
        x: 431
        y: 173
        visible: false
        source: "images/number-5.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: six
        x: 535
        y: 164
        width: 900
        visible: false
        source: "images/six.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: seven
        x: 203
        y: 223
        visible: false
        source: "images/seven.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: number8
        x: 101
        y: 379
        visible: false
        source: "images/number-8.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: number9
        x: 328
        y: 428
        visible: false
        source: "images/number-9.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: zero
        x: 265
        y: 366
        visible: false
        source: "images/zero.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: telephone
        x: 468
        y: 366
        visible: false
        anchors.verticalCenter: parent.verticalCenter
        source: "images/telephone.png"
        anchors.verticalCenterOffset: 85
        anchors.horizontalCenterOffset: 151
        anchors.horizontalCenter: parent.horizontalCenter
        fillMode: Image.PreserveAspectFit
    }

    states: [
        State {
            name: "musicState"

            PropertyChanges {
                target: backbutton
                x: 290
                y: 570
                width: 180
                height: 80
                visible: true

            }

            PropertyChanges {
                target: nextbutton
                x: 763
                y: 570
                width: 180
                height: 80
                visible: true

            }

            PropertyChanges {
                target: pause
                x: 597
                y: 570
                width: 180
                height: 80
                visible: true
                z: 0

            }

            PropertyChanges {
                target: play
                x: 442
                y: 570
                width: 180
                height: 80
                opacity: 1
                visible: true
                z: 0

            }
            PropertyChanges {
                target: map
                x: 0
                y: 0
                width: 900
                height: 530
                opacity: 0.5
                visible: false
            }

            PropertyChanges {
                target: calendar
                visible: false
            }

            PropertyChanges {
                target: clock
                visible: false
            }

            PropertyChanges {
                target: text1
                visible: false
            }

            PropertyChanges {
                target: text2
                visible: false
            }

            PropertyChanges {
                target: textInput
                visible: false
            }

            PropertyChanges {
                target: textInput1
                visible: false
            }

            PropertyChanges {
                target: text3
                visible: false
            }

            PropertyChanges {
                target: displayArea
                visible: true
                color: "#404244"

            }
            PropertyChanges {
                target: animatedGif
                width: 450
                height: 450
                visible:true
                anchors.verticalCenterOffset: -64
                anchors.horizontalCenterOffset: 0
                baselineOffset: 0
                antialiasing: true
                scale: 1
                transformOrigin: Item.Center
                rotation: 180
                paused: true
                anchors.bottomMargin: 0
                playing: false
                speed: 0.5
                fillMode: Image.Stretch

            }

            PropertyChanges {
                target: logo
                visible: false
            }

            PropertyChanges {
                target: slider
                x: 996
                y: 180
                width: 300
                height: 60
                visible: true
                rotation: 90
            }

            PropertyChanges {
                target: audioProgress
                x: 0
                y: 530
                width: 50
                visible: true
                color: "#141bb1"
                radius: audioProgress.radius =  progressRec.radius
                border.width: 1
            }

            PropertyChanges {
                target: progressRec
                x: 368
                y: 530
                width: 500
                height: 8
                visible: true
                radius: 5
                border.color: "#000000"
                border.width: 1
                rotation: 0
                z: 1
            }

            PropertyChanges {
                target: progressRec
                color: "#ffffff"
            }



            PropertyChanges {
                target: tumblerHour
                x: 691
                y: 140
            }
        },
        State {
            name: "navigationState"

            PropertyChanges { target: mapItem; visible: true }

            PropertyChanges {
                target: backbutton
                x: 20
                y: 329
                width: 200
                height: 153
                visible: false

            }

            PropertyChanges {
                target: nextbutton
                x: 680
                y: 329
                width: 200
                height: 153
                visible: false

            }

            PropertyChanges {
                target: pause
                x: 450
                y: 323
                width: 200
                height: 147
                visible: false

            }

            PropertyChanges {
                target: play
                x: 250
                y: 329
                width: 200
                height: 141
                visible: false
            }

            PropertyChanges {
                target: text1
                visible: false
            }

            PropertyChanges {
                target: text2
                visible: false
            }

            PropertyChanges {
                target: textInput
                visible: false
            }

            PropertyChanges {
                target: clock
                visible: false
            }

            PropertyChanges {
                target: calendar
                visible: false
            }

            PropertyChanges {
                target: textInput1
                visible: false
            }

            PropertyChanges {
                target: text3
                visible: false
            }

            PropertyChanges {
                target: logo
                visible: false
            }

            PropertyChanges {
                target: darkMode
                layer.enabled: false
                focus: true
            }
        },
        State {
            name: "settingState"

            PropertyChanges {
                target: text1
                x: 174
                y: 126
                width: 136
                height: 40
                visible: true
                color: "#ffffff"
                text: qsTr("set time")
                font.pixelSize: 24
                font.bold: true
                font.family: "Arial"
            }

            PropertyChanges {
                target: displayArea
                color: "#404244"
            }

            PropertyChanges {
                target: text2
                x: 174
                y: 286
                width: 130
                height: 44
                visible: true
                color: "#ffffff"
                text: qsTr("set date")
                font.pixelSize: 20
                font.bold: true
            }

            PropertyChanges {
                target: textInput
                x: 437
                y: 153
                width: 272
                height: 40
                visible: false
                color: "#ffffff"
                text: qsTr("time Input")
                font.pixelSize: 18
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }

            PropertyChanges {
                target: clock
                x: 87
                y: -29
                width: 130
                height: 93
                visible: false
                z: 0
            }

            PropertyChanges {
                target: calendar
                x: 243
                y: 241
                width: 113
                height: 70
                visible: false
            }

            PropertyChanges {
                target: pause
                x: 497
                y: 400
            }

            PropertyChanges {
                target: play
                x: 87
                y: 634
            }

            PropertyChanges {
                target: map
                x: 243
                y: 347
            }

            PropertyChanges {
                target: textInput1
                x: 437
                y: 257
                width: 272
                height: 40
                visible: false
                color: "#ffffff"
                text: qsTr("date Input")
                font.pixelSize: 18
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }

            PropertyChanges {
                target: text3
                x: 162
                y: 23
                width: 609
                height: 67
                visible: true
                color: "#ffffff"
                text: qsTr("Setting")
                font.pixelSize: 25
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                font.styleName: "Bold"
                font.family: "Arial"
                font.bold: true
            }

            PropertyChanges {
                target: logo
                visible: false
            }

            PropertyChanges {
                target: darkMode
                x: 497
                y: 408
                width: 159
                height: 66
                visible: true
                text: qsTr("Dark Mode")
                font.pointSize: 8
                display: AbstractButton.TextBesideIcon
                autoRepeat: false
                checked: false
            }

            PropertyChanges {
                target: darkMode
                font.pointSize: 11

//                ColorAnimation {
//                    from: "white"
                //                    to: "black"
                //                    duration: 200
                //                }
            }

            PropertyChanges {
                target: textDarkMode
                x: 174
                y: 417
                visible: true
                color: "#ffffff"
                text: qsTr("Dark Mode")
                font.pixelSize: 20
                font.bold: true
            }

            PropertyChanges {
                target: tumblerHour
                x: 431
                y: 96
                width: 40
                height: 100
                visible: true
                font.pointSize: 12
            }

            PropertyChanges {
                target: tumblerMinute
                x: 530
                y: 96
                width: 40
                height: 100
                visible: true
                font.pointSize: 12
            }

//            PropertyChanges {
//                target: tumbler
//                x: 628
//                y: 123
//            }

            PropertyChanges {
                target: tumblerPM
                x: 628
                y: 96
                width: 40
                height: 100
                visible: true
            }

            PropertyChanges {
                target: tumblerDay
                x: 431
                y: 263
                width: 40
                height: 100
                visible: true
            }

            PropertyChanges {
                target: tumblerMonth
                x: 530
                y: 263
                width: 40
                height: 100
                visible: true
            }

            PropertyChanges {
                target: tumblerYear
                x: 628
                y: 263
                width: 40
                height: 100
                visible: true
            }
        },
        State {
            name: "phoneState"

            PropertyChanges {
                target: backbutton
                visible: false
            }

            PropertyChanges {
                target: nextbutton
                visible: false
            }

            PropertyChanges {
                target: pause
                visible: false
            }

            PropertyChanges {
                target: play
                visible: false
            }

            PropertyChanges {
                target: map
                visible: false
            }

            PropertyChanges {
                target: text1
                visible: false
            }

            PropertyChanges {
                target: text2
                visible: false
            }

            PropertyChanges {
                target: textInput
                visible: false
            }

            PropertyChanges {
                target: clock
                visible: false
            }

            PropertyChanges {
                target: calendar
                visible: false
            }

            PropertyChanges {
                target: textInput1
                visible: false
            }

            PropertyChanges {
                target: text3
                y: 69
                visible: false
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignTop
                anchors.horizontalCenterOffset: 1
            }

            PropertyChanges {
                target: numberone
                x: 407
                y: 169
                width: 100
                height: 80
                visible: true
            }

            PropertyChanges {
                target: number2
                x: 557
                y: 169
                width: 100
                height: 80
                visible: true
            }

            PropertyChanges {
                target: number3
                x: 707
                y: 169
                width: 100
                height: 80
                visible: true
            }

            PropertyChanges {
                target: numberfour
                x: 407
                y: 289
                width: 100
                height: 80
                visible: true
            }

            PropertyChanges {
                target: number5
                x: 557
                y: 289
                width: 100
                height: 80
                visible: true
            }

            PropertyChanges {
                target: six
                x: 707
                y: 289
                width: 100
                height: 80
                visible: true
            }

            PropertyChanges {
                target: seven
                x: 407
                y: 419
                width: 100
                height: 80
                visible: true
            }

            PropertyChanges {
                target: number8
                x: 557
                y: 419
                width: 100
                height: 80
                visible: true
            }

            PropertyChanges {
                target: number9
                x: 707
                y: 419
                width: 100
                height: 80
                visible: true
            }

            PropertyChanges {
                target: zero
                x: 557
                y: 529
                width: 100
                height: 80
                visible: true
            }

            PropertyChanges {
                target: telephone
                x: 550
                y: 420
                width: 100
                height: 80
                visible: true
                anchors.verticalCenterOffset: 220
                anchors.horizontalCenterOffset: 139
            }

            PropertyChanges {
                target: text3
                width: 307
                height: 34
                visible: true
                color: "#ffffff"
                text: qsTr(" _ _ _ _ _ _ _ _ ")
                font.pixelSize: 15
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }

            PropertyChanges {
                target: displayArea
                color: "#404244"
            }

            PropertyChanges {
                target: logo
                visible: false
            }
        }

    ]
    Map {
        id: mapItem
        visible: false
        color: "#ffffff"
        anchors.fill: parent

        plugin: Plugin {
            name: "osm"
        }

        center: QtPositioning.coordinate(31.2565, 32.2840)

        zoomLevel: 13.5
    }

    Image {
        id: logo
        anchors.fill: parent
        source: "images/logo.png"
        sourceSize.height: 0
        sourceSize.width: 0
        fillMode: Image.PreserveAspectFit
    }

    Slider {
        id: slider
        x: 996
        y: 244
        visible: false
        anchors.verticalCenter: parent.verticalCenter
        anchors.verticalCenterOffset: 0
        focus: false
        from: 1  // Set the minimum value of the Slider
        to: 0

        value: mediaPlayer.volume // Initialize the Slider value with the initial volume

                // When the slider value changes, update the MediaPlayer volume
                onValueChanged: {
                    mediaPlayer.volume = slider.value
                    startCam.focus = true
                }
    }

    Switch {
        id: darkMode
        x: 276
        y: 382
        visible: false
        text: qsTr("Dark Mode")
        layer.enabled: false

        onCheckedChanged: {
                if (darkMode.checked) {
                    root.color="dark";
                    displayArea.color="dark";
                    nightMode.visible=true;


                } else {
                    root.color="#404244";
                    displayArea.color="#2E2F30";
                       nightMode.visible=false
                }
            }

    }

    Text {
        id: textDarkMode
        x: 164
        y: 388
        visible: false
        text: qsTr("Text")
        font.pixelSize: 12
    }

    Tumbler {
        id: tumblerHour
        x: 704
        y: 113
        visible: false
        model: 12
        background: Rectangle {
            color: "white"
        }
    }

    Tumbler {
        id: tumblerMinute
        x: 530
        y: 109
        visible: false
        model: 60
        background: Rectangle {
            color: "white"
        }
    }

    Tumbler {
        id: tumblerDay
        x: 421
        y: 257
        visible: false
        model: 31
        background: Rectangle {
            color: "white"
        }
    }

    Tumbler {
        id: tumblerMonth
        x: 530
        y: 257
        visible: false
        model: 12
        background: Rectangle {
            color: "white"
        }
    }

    Tumbler {
        id: tumblerYear
        x: 640
        y: 257
        visible: false
        model: ListModel {
                ListElement { value: 2024 }
                ListElement { value: 2025 }
                ListElement { value: 2026 }
                ListElement { value: 2027 }
                ListElement { value: 2028 }
                ListElement { value: 2029 }
                ListElement { value: 2030 }
            }
        background: Rectangle {
            color: "white"
        }
    }

    Tumbler {
        id: tumblerPM
        x: 639
        y: 149
        visible: false
        model: ["AM", "PM"]
        background: Rectangle {
            color: "white"
        }
    }
}



/*##^##
Designer {
    D{i:0;formeditorZoom:0.5}D{i:11}D{i:115}D{i:117}
}
##^##*/
