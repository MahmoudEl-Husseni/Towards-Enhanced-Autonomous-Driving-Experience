import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Extras 1.4
import Qt.labs.calendar 1.0
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.12
import QtPositioning 5.12
import QtLocation 5.12
import QtQuick.Controls 2.12
import QtMultimedia 5.12
import com.company.serialmanager 1.0

Window {
    id: root
    maximumHeight: 768
    maximumWidth: 1360
    minimumHeight: 768
    minimumWidth: 1360
    width: 1360
    height: 768
    visible: true
    title: qsTr("Infotainment System")
    color: "#2E2F30"
    property string gear: "P";
    MediaPlayer {
        id: mediaPlayer
        source: "/file.mp3"
         volume: 0.5

    }
    Connections {
        target: serialManager
        onTemperatureChanged: {text3.text = ""+newTemp+" C";}
        onSelectedGearChanged:{
            if(selectedGear=="R"){
                camera.start() // Start the camera
                viewfinder.visible = true // Show the viewfinder
            }else{
                camera.stop() // Stop the camera (not start)
                viewfinder.visible = false // Hide the viewfinder
            }

            gear=selectedGear;
        }
    }

    VideoOutput {
            id: viewfinder
            x: 124
            y: 70
            width: 1236
            height: 698
//            anchors.fill: parent
//            anchors.centerIn: parent
            source: camera
            z: 1
            autoOrientation: true
            visible: false

          }

          Camera {
            id: camera

          }



    Item {
        id: statusBar
        x: 0
        y: 0
        width: 1360
        height: 70
        Image {
            id: bluetooth
            x: 1069
            y: 12
            width: 151
            height: 50
            source: "images/bluetooth.png"
            fillMode: Image.PreserveAspectFit
        }

        Image {
            id: weather
            x: 732
            y: 8
            width: 103
            height: 54
            source: "images/weather.png"
            fillMode: Image.PreserveAspectFit
        }

        Timer {
            interval: 1000 // Update every 1000 milliseconds (1 second)
            running: true
            repeat: true

            onTriggered: {
                // Update the displayed time every second
                timeText.text =  Qt.formatDateTime(new Date(), "h:mm ap")
            }
        }
        Text {
            id: timeText
            x: 1216
            y: 12
            width: 119
            height: 50
            color: "#ffffff"
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            textFormat: Text.RichText
            font.italic: false
            font.bold: true
            font.family: "Arial"
        }

        Timer {
                    interval: 1000 // Update every 1000 milliseconds (1 second)
                    running: true
                    repeat: true

                    onTriggered: {
                        // Update the displayed date every second
                        dateText.text = Qt.formatDateTime(new Date(), "dd/MM/yyyy")
                    }
                }
        Text {
            id: dateText
            x: 22
            y: 22
            width: 169
            height: 31
            color: "#ffffff"
            font.pixelSize: 25
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.styleName: "Italic"
            font.bold: true
            font.family: "Arial"
        }

        Text {
            id: text3
            x: 809
            y: 14
            width: 96
            height: 39
            color: "#ffffff"
            text: qsTr("24 C")
            font.pixelSize: 24
            horizontalAlignment: Text.AlignLeft
            verticalAlignment: Text.AlignVCenter
            font.bold: true
            font.family: "Tahoma"
        }

        Image {
            id: signal
            x: 942
            width: 71
            height: 57
            anchors.verticalCenter: parent.verticalCenter
            source: "images/signal.png"
            anchors.verticalCenterOffset: -1
            sourceSize.width: 510
            fillMode: Image.PreserveAspectFit
        }

        Image {
            id: nightMode
            x: 1019
            width: 71
            height: 57
            anchors.verticalCenter: parent.verticalCenter
            source: "images/nightMode.png"
            anchors.verticalCenterOffset: -1
            sourceSize.width: 510
            fillMode: Image.PreserveAspectFit
            visible: false
        }
    }

    Item {
        id: icons
        x: 0
        y: 70
        width: 136
        height: 692
        Image {
            id: music
            x: 6
            y: 125
            width: 122
            height: 75
            source: "images/music.png"
            fillMode: Image.PreserveAspectFit

            Text {
                id: text4
                x: 19
                y: 68
                width: 85
                color: "#ffffff"
                text: qsTr("Audio")
                font.pixelSize: 20
                horizontalAlignment: Text.AlignHCenter
                font.bold: true
            }

            MouseArea {
                id: mouseArea2
                anchors.fill: parent

                     // onClicked:  mediaPlayer.play()
                    onClicked: displayArea.state = "musicState"

            }
        }

        Image {
            id: navigation
            x: 18
            y: 8
            width: 100
            height: 60
            source: "images/navigation.png"
            fillMode: Image.PreserveAspectFit

            MouseArea {
                id: mouseArea1
                anchors.fill: parent
                anchors.rightMargin: 0
                anchors.bottomMargin: -26
                anchors.leftMargin: 0
                anchors.topMargin: 26
                antialiasing: false
                focus: true


                onClicked: displayArea.state = "navigationState"

            }

            Text {
                id: text5
                x: 4
                y: 66
                width: 92
                height: 20
                color: "#ffffff"
                text: qsTr("Navigation")
                font.pixelSize: 20
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignBottom
                font.bold: true
                font.family: "Arial"
            }
        }

        Image {
            id: settings
            x: 18
            y: 555
            width: 100
            height: 64
            source: "images/settings.png"
            fillMode: Image.PreserveAspectFit

            Text {
                id: text6
                x: 9
                y: 68
                color: "#ffffff"
                text: qsTr("Settings")
                font.pixelSize: 20
                horizontalAlignment: Text.AlignHCenter
                font.bold: true
            }

            MouseArea {
                id: mouseArea5
                x: 0
                y: 0
                width: 100
                height: 92


                    onClicked: displayArea.state = "settingState"

            }
        }
        Image {
            id: telephone
            x: 32
            y: 401
            width: 73
            height: 69
            source: "images/telephone.png"
            fillMode: Image.PreserveAspectFit

            Text {
                id: text7
                x: 7
                y: 73
                color: "#ffffff"
                text: qsTr("Phone")
                font.pixelSize: 20
                horizontalAlignment: Text.AlignHCenter
                font.bold: true
            }

            MouseArea {
                id: mouseArea4
                x: -13
                y: -3
                width: 100
                height: 100
                focus: true


                    onClicked: displayArea.state = "phoneState"

            }
        }

        Image {
            id: thermostat
            x: 18
            y: 258
            width: 100
            height: 80
            source: "images/thermostat.png"
            fillMode: Image.PreserveAspectFit

            Text {
                id: text8
                x: -2
                y: 76
                color: "#ffffff"
                text: qsTr("AC control")
                font.pixelSize: 20
                horizontalAlignment: Text.AlignHCenter
                font.bold: true
            }

            MouseArea {
                id: mouseArea3
                width: 100
                height: 100
            }
        }

    }

    MyDisplayArea {
        id: displayArea
        x: 124
        y: 70
        width: 1236
        height: 698
        color: "#2E2F30"
    }


}


/*##^##
Designer {
    D{i:0;formeditorZoom:0.5}
}
##^##*/
