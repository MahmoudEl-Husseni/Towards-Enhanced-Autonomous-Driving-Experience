import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Extras 1.4
import Qt.labs.calendar 1.0
import QtQuick.Controls 2.3
import com.company.serialmanager 1.0

Window {
    maximumHeight: 494
    maximumWidth: 1300
    minimumHeight: 494
    minimumWidth: 1300
    width: 1300
    height: 494
    visible: true
    title: qsTr("Dashboard")
    color: "#202020"
    property int currentspeedValue: 0

//    Image {
//        id: image1
//        anchors.fill: parent
//        source: "images/qit91mty-5 - Copy.png"
//        fillMode: Image.PreserveAspectFit
//    }


    Image {
        id: image
        anchors.fill: parent
        source: "images/qit91mty-1 - Copy (2).png"
        fillMode: Image.PreserveAspectFit
    }

//    Timer {
//            id: timer
//            interval: 125 // Update every 1000 milliseconds (1 second)
//            running: true
//            repeat: true
//            onTriggered: {

//                // Increment the current value
//                currentspeedValue += 1

//                // Update the label text
//                speed_read.text = currentspeedValue

//                // Stop the timer when the value reaches 260
//                if (currentspeedValue >= 260) {
//                    timer.stop()
//                }
//            }
//        }


    Text {
        id: speed_read
        x: 147
        y: 199
        width: 172
        height: 97
        color: "#ffffff"
        text: "0"
        font.pixelSize: 85
        horizontalAlignment: Text.AlignHCenter
        font.weight: Font.Black
    }

    Text {
        id: text1
        x: 974
        y: 191
        width: 167
        height: 102
        color: "#ffffff"
        text: qsTr("P")
        font.pixelSize: 100
        horizontalAlignment: Text.AlignHCenter
        font.weight: Font.Black
    }



    Label {
        id: tempLabel
        x: 757
        y: 10
        color: "#ffffff"
        text: "N/A"
        verticalAlignment: Text.AlignVCenter
        font.styleName: "Black"
        font.family: "Arial"
        font.pointSize: 20
        font.weight: Font.Black
        font.bold: false
    }



    Label {
        id: label1
        x: 815
        y: 17
        color: "#ffffff"
        text: qsTr("C")
        verticalAlignment: Text.AlignVCenter
        font.styleName: "Black"
        font.family: "Arial"
        font.pointSize: 15
        font.bold: false
        font.weight: Font.Black
    }

    Label {
        id: label2
        x: 808
        y: 9
        color: "#ffffff"
        text: qsTr("O")
        verticalAlignment: Text.AlignVCenter
        font.styleName: "Black"
        font.family: "Arial"
        font.pointSize: 8
        font.bold: false
        font.weight: Font.Medium
    }

    Image {
        id: seatbelt
        x: 483
        y: 94
        source: "images/seat-belt.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: lowbeam
        x: 730
        y: 157
        width: 44
        height: 44
        source: "images/low-beam.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: speedlimit
        x: 528
        y: 157
        width: 43
        height: 44
        source: "images/speed-limit.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: highbeam
        x: 786
        y: 95
        width: 40
        height: 41
        source: "images/high-beam.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: assistdisable
        x: 628
        y: 159
        width: 44
        height: 44
        source: "images/assist-disable.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: steeringerror
        x: 684
        y: 94
        width: 41
        height: 42
        source: "images/steering-error.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: lowbat
        x: 577
        y: 94
        width: 41
        height: 42
        source: "images/Low-bat.png"
        fillMode: Image.PreserveAspectFit
    }


    Text {
        id: text2
        x: 974
        y: 309
        width: 167
        height: 35
        color: "#7f7f7f"
        text: qsTr("P N D R")
        font.pixelSize: 25
        horizontalAlignment: Text.AlignHCenter
        font.weight: Font.Black
    }

    //Upper-icons

    Image {
        id: parkingbreak
        x: 535
        y: 10
        source: "images/Parking-break.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: dooropen
        x: 603
        y: 10
        width: 39
        height: 42
        visible: true
        source: "images/door-open.png"
        fillMode: Image.PreserveAspectFit
    }

    Image {
        id: rightlabel
        x: 660
        y: 11
        width: 37
        height: 40
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
        x: 487
        y: 11
        width: 37
        height: 40
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
        onSpeedChanged:{speed_read.text = ""+newSpeed;}
        onDoorStatusChanged:{
            if(newDoorStatus){
                dooropen.visible=false;
            }else{
                dooropen.visible=true;
            }
        }
    }

}


/*##^##
Designer {
    D{i:0;formeditorZoom:1.75}
}
##^##*/
