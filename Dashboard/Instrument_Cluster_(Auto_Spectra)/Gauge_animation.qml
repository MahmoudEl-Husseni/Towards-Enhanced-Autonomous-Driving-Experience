import QtQuick 2.15
import QtQuick.Controls 2.0
import QtQuick.Controls.Styles 1.4
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtGraphicalEffects 1.0

CircularGauge {
    id: gauge

    style: CircularGaugeStyle {
        minimumValueAngle: -150
        maximumValueAngle: 150

        background: Rectangle {
            implicitHeight: gauge.height
            implicitWidth: gauge.width
            color:  "#0039383c"
            anchors.centerIn: parent
            radius: 360
            Image {
                id:needleColor1
                visible: false
                anchors.fill: parent
                source: "/img/background60.svg"
                asynchronous: true
                sourceSize {
                    width: width
                }
            }
            Image {
                id:needleColor2
                visible: false
                anchors.fill: parent
                source: "/img/background120.svg"
                asynchronous: true
                sourceSize {
                    width: width
                }
            }
            Image {
                id:needleColor3
                visible: false
                anchors.fill: parent
                source: "/img/background.svg"
                asynchronous: true
                sourceSize {
                    width: width
                }
            }
            // Add tick labels behind the Canvas
            Repeater {
                model: 7
                delegate: Text {
                    property real angle: -200 + (index * 37)
                    property real radius: gauge.width / 2 - 25 // Adjust radius as needed
                    property real xOffset: Math.cos(Math.PI * angle / 180) * radius
                    property real yOffset: Math.sin(Math.PI * angle / 180) * radius
                    x: (gauge.width / 2) + xOffset - width / 2
                    y: (gauge.height / 2) + yOffset - height / 2
                    text: (20+(index *20)).toFixed(0)
                    color: "white"
                    font.pixelSize: 20
                    transform: Rotation {
                        origin.x: x + width / 2
                        origin.y: y + height / 2
                        angle: angle
                    }
                }
            }

            Canvas {
                property int value: gauge.value

                anchors.fill: parent
                onValueChanged: requestPaint()

                function degreesToRadians(degrees) {
                    return degrees * (Math.PI / 180);
                }
                Repeater {
                    z:1
                    model: 7
                    delegate: Text {
                        property real angle: -200 + (index * 37)
                        property real radius: gauge.width / 2 - 25 // Adjust radius as needed
                        property real xOffset: Math.cos(Math.PI * angle / 180) * radius
                        property real yOffset: Math.sin(Math.PI * angle / 180) * radius
                        x: (gauge.width / 2) + xOffset - width / 2
                        y: (gauge.height / 2) + yOffset - height / 2
                        text: (20+(index *20)).toFixed(0)
                        color: "white"
                        font.pixelSize: 20
                        transform: Rotation {
                            origin.x: x + width / 2
                            origin.y: y + height / 2
                            angle: angle
                        }
                    }
                }
                onPaint: {
                    if(gauge.value>0&&gauge.value<60){
                        needleColor1.visible=true
                        needleColor2.visible=false
                        needleColor3.visible=false
                    }
                    if (gauge.value>=60&&gauge.value<120){
                        needleColor1.visible=false
                        needleColor2.visible=true
                        needleColor3.visible=false
                    }
                    if(gauge.value>=120){
                        needleColor1.visible=false
                        needleColor2.visible=false
                        needleColor3.visible=true
                    }
                    var ctx = getContext("2d");
                    ctx.reset();
                    ctx.beginPath();
                    ctx.strokeStyle = "#39383c"
                    ctx.lineWidth = 46
                    ctx.arc(outerRadius,
                        outerRadius,
                        outerRadius - ctx.lineWidth / 2,
                        degreesToRadians(valueToAngle(gauge.value) - 90),
                        degreesToRadians(valueToAngle(gauge.maximumValue + 1) - 90)
                    );
                    ctx.stroke();
                }
            }
        }


        foreground: Item {
            Text {
                x: 118
                y: 200
                width: 167
                height: 97
                font.pixelSize: 80
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignBottom
                font.family: "Futura"
                font.bold: true
                smooth: true
                anchors.centerIn: parent
                text: gauge.value.toFixed(0)
                color: "white"
                visible: false
            }
        }

        tickmarkLabel: Text {
            visible: false
        }

        tickmark: Rectangle {
            visible: false
        }
        needle: Item {
            visible: false
        }
        minorTickmark: Rectangle {
            visible: false
        }
    }
}

/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
##^##*/
