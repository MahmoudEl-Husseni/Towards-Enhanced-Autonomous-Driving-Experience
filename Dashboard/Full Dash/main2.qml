import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Extras 1.4
import Qt.labs.calendar 1.0
import com.company.serialmanager 1.0

Window {
    id: window
    maximumHeight: 768
    maximumWidth: 1360
    minimumHeight: 768
    minimumWidth: 1360
    width: 1360
    height: 768
    visible: true
//    visibility: "FullScreen"
//    screen: Qt.application.screens[1]
    title: qsTr("Instrument Cluster")
    color: "#3b3b3a"


    Image {
        id: image
        width: 1431
        height: 808
        opacity: 0.697
        anchors.verticalCenter: parent.verticalCenter
        source: "images/dark-mode.png"
        anchors.verticalCenterOffset: 15
        anchors.horizontalCenterOffset: 1
        anchors.horizontalCenter: parent.horizontalCenter
        asynchronous: true
        mipmap: true
        mirror: false
        smooth: true
        cache: true
        autoTransform: true
        clip: false

    }

//    Timer {
//        interval: 1000
//        running: true
//        repeat: false
//        property bool isDaytime: false
//        onTriggered: {
//            var currentHour = new Date().getHours()
//            // Assuming day time is from 6 AM to 6 PM
//            var isDaytime = currentHour >= 6 && currentHour < 18
//            image.source = isDaytime ? "images/light-mode.png" : "images/dark-mode.png"
//        }
//    }

    Text {
        id: _caution
        x: 497
        y: 301
        color: "#d92a27"
        text: qsTr("Caution:")
        font.pixelSize: 21
        verticalAlignment: Text.AlignVCenter
        font.styleName: "ExtraBold"
        font.family: "Open Sans"
        font.weight: Font.Bold
    }

    Text {
        id: _cautionMassage
        x: 594
        y: 303
        width: 273
        height: 25
        color: "#ffffff"
        text: qsTr("You need to take a Break")
        font.pixelSize: 21
        horizontalAlignment: Text.AlignLeft
        verticalAlignment: Text.AlignVCenter
        font.family: "MS Shell Dlg 2"
        font.weight: Font.Black
    }

    Text {
        id: speed_read
        x: 139
        y: 346
        width: 199
        height: 106
        color: "#ffffff"
        text: "0"
        font.pixelSize: 110
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        lineHeight: 0
        focus: false
        font.family: "MS Shell Dlg 2"
        font.weight: Font.Black
    }

    Text {
        id: selected_gear
        x: 1016
        y: 329
        width: 202
        height: 123
        color: "#ffffff"
        text: qsTr("P")
        font.pixelSize: 125
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        font.weight: Font.Black
    }



    Text {
        id: tempLabel
        x: 838
        y: 150
        width: 61
        height: 21
        color: "#ffffff"
        text: "N/A"

        horizontalAlignment: Text.AlignRight
        verticalAlignment: Text.AlignVCenter
        font.styleName: "ExtraBold"
        font.family: "Open Sans"
        font.pointSize: 20
        font.weight: Font.Black
        font.bold: false
    }



    Text {
        id: label2
        x: 906
        y: 149
        height: 11
        color: "#b7b7b7"
        text: qsTr("O")
        verticalAlignment: Text.AlignVCenter
        font.styleName: "Light"
        font.family: "Open Sans"
        font.pointSize: 6
        font.bold: false
        font.weight: Font.Medium
    }



    Image {
        id: speedlimit
        x: 803
        y: 227
        width: 50
        height: 50
        source: "images/Set 1/06.png"
        fillMode: Image.PreserveAspectFit
    }


    Image {
        id: assistdisable
        x: 705
        y: 230
        width: 50
        height: 50
        source: "images/assist-disable.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: steeringerror
        x: 506
        y: 232
        width: 50
        height: 50
        source: "images/steering-error.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: lowbat
        x: 603
        y: 230
        width: 50
        height: 50
        source: "images/Low-bat.png"
        fillMode: Image.PreserveAspectFit
    }



    //Upper-icons

    Image {
        id: parkingbreak
        x: 609
        y: 136
        width: 50
        height: 50
        source: "images/Parking-break.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: dooropen
        x: 693
        y: 136
        width: 50
        height: 50
        visible: true
        source: "images/door-open.png"
        fillMode: Image.PreserveAspectFit
    }

    Text {
        id: clockdate
        x: 413
        y: 150
        width: 83
        height: 21
        color: "#ffffff"
        text: "00:00"
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        font.pointSize: 20
        font.family: "Open Sans"
        font.styleName: "ExtraBold"
        font.weight: Font.Black
        // Display current time
    }
    Text {
        id: clockDateSign
        x: 497
        y: 155
        width: 22
        height: 18
        color: "#b7b7b7"
        text: "AM"
        font.pixelSize: 15
        verticalAlignment: Text.AlignTop
        font.family: "Open Sans"
        font.styleName: "Light"
        // Display current time
    }
        // Update every second
        Timer {
            interval: 1000
            running: true
            repeat: true
            onTriggered: {
                clockdate.text = formatTimeWithoutAMPM(new Date())
                clockDateSign.text = Qt.formatTime(new Date(), "AP")
            }
            function formatTimeWithoutAMPM(dateTime) {
                var hours = dateTime.getHours()
                var minutes = dateTime.getMinutes()

                // Convert to 12-hour format
                var ampm = hours >= 12 ? "PM" : "AM"
                hours = hours % 12
                hours = hours ? hours : 12  // Handle midnight (12:00 AM)

                // Add leading zeros
                hours = ("0" + hours).slice(-2)
                minutes = ("0" + minutes).slice(-2)

                return hours + ":" + minutes
            }
        }


    Image {
        id: rightlabel
        x: 761
        y: 144
        width: 40
        height: 37
        source: "images/right.png"
        fillMode: Image.PreserveAspectFit
        opacity:0
        visible: true

        SequentialAnimation {
            running: true
            loops: Animation.Infinite  // Infinite looping

            PropertyAction {
                target: rightlabel
                property: "opacity"
                value: 1
            }

            NumberAnimation {
                target: rightlabel
                property: "opacity"
                to: 0
                duration: 600  // Adjust the duration of the fade-out
            }

            NumberAnimation {
                target: rightlabel
                property: "opacity"
                to: 1
                duration: 600  // Adjust the duration of the fade-in
            }
        }
    }

    Image {
        id: leftlabel
        x: 548
        y: 144
        width: 40
        height: 37
        source: "images/left.png"
        fillMode: Image.PreserveAspectFit
        opacity:0

        SequentialAnimation {
            running: true
            loops: Animation.Infinite  // Infinite looping

            PropertyAction {
                target: leftlabel
                property: "opacity"
                value: 1
            }

            NumberAnimation {
                target: leftlabel
                property: "opacity"
                to: 0
                duration: 600  // Adjust the duration of the fade-out
            }

            NumberAnimation {
                target: leftlabel
                property: "opacity"
                to: 1
                duration: 600  // Adjust the duration of the fade-in
            }
        }
    }

    Connections {
        target: serialManager
        onTemperatureChanged: {tempLabel.text = ""+newTemp;}
        onSpeedChanged:{if(selected_gear.text === "P"){
                speed_read.text = 0;
            }else{
            speed_read.text = ""+newSpeed;}
        }
        onDoorStatusChanged:{
            if(newDoorStatus){
                dooropen.visible=false;
            }else{
                dooropen.visible=true;
            }
        }
        onSelectedGearChanged:{
            selected_gear.text=selectedGear;
            if (selected_gear.text === "P") {
                _P.color = "#ffffff";
            } else {
                _P.color = "#7f7f7f";
            }
            if (selected_gear.text === "N") {
                _N.color = "#ffffff";
            } else {
                _N.color = "#7f7f7f";
            }
            if (selected_gear.text === "D") {
                _D.color = "#ffffff";
            } else {
                _D.color = "#7f7f7f";
            }
            if (selected_gear.text === "R") {
                _R.color = "#ffffff";
            } else {
                _R.color = "#7f7f7f";
            }
        }
    }


    Text {
        id: _P
        x: 1047
        y: 468
        width: 31
        height: 35
        color: "#ffffff"
        text: qsTr("P")
        font.pixelSize: 30
        horizontalAlignment: Text.AlignHCenter
        font.weight: Font.Black
    }

    Text {
        id: _N
        x: 1084
        y: 468
        width: 31
        height: 35
        color: "#7f7f7f"
        text: qsTr("N")
        font.pixelSize: 30
        horizontalAlignment: Text.AlignHCenter
        font.weight: Font.Black
    }

    Text {
        id: _D
        x: 1119
        y: 468
        width: 31
        height: 35
        color: "#7f7f7f"
        text: qsTr("D")
        font.pixelSize: 30
        horizontalAlignment: Text.AlignHCenter
        font.weight: Font.Black
    }

    Text {
        id: _R
        x: 1155
        y: 468
        width: 31
        height: 35
        color: "#7f7f7f"
        text: qsTr("R")
        font.pixelSize: 30
        horizontalAlignment: Text.AlignHCenter
        font.weight: Font.Black
    }

    Text {
        id: label3
        x: 911
        y: 156
        width: 14
        height: 15
        color: "#b7b7b7"
        text: qsTr("C")
        verticalAlignment: Text.AlignVCenter
        font.pointSize: 11
        font.weight: Font.Medium
        font.styleName: "Regular"
        font.family: "Open Sans"
        font.bold: false
    }

    Image {
        id: image1
        x: 445
        y: 665
        width: 454
        height: 103
        opacity: 0.5
        source: "images/base_1.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: highbeam
        x: 548
        y: 679
        width: 75
        height: 75
        source: "images/high-beam.png"
        fillMode: Image.PreserveAspectFit
    }
    Image {
        id: lowbeam
        x: 733
        y: 679
        width: 75
        height: 75
        source: "images/low-beam.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: image2
        x: 643
        y: 679
        width: 75
        height: 75
        source: "images/Adaptive-on.png"
        fillMode: Image.PreserveAspectFit
    }
}






/*##^##
Designer {
    D{i:0;formeditorZoom:0.66}D{i:2;invisible:true;locked:true}D{i:3;invisible:true}
}
##^##*/
