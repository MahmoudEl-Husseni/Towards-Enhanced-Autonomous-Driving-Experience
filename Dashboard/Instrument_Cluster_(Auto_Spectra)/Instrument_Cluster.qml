import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Extras 1.4
import QtMultimedia 5.15
import QtGraphicalEffects 1.0
import QtQuick.Controls.Styles 1.4
import com.company.serialmanager 1.0
import MyPythonScript 1.0
import com.company.cardatareceiver 1.0
import spotifyreceiver 1.0
Item {
    id: item1
    property int hasRealData: 0
    // Rectangle{
    //     width: 1024
    //     height: 600
    //     color: "#1e1e1e"

    // }

    Image {
        id: color_fill_1
        source: "images/color_fill_1.png"
        x: 0
        y: 0
        opacity: 0.8
    }


    Image {
        id: rectangle_1
        source: "images/rectangle_1.png"
        x: 200
        y: 153
        opacity: 1
    }




    Rectangle {
        id: artistImage
        radius: 8
        visible: false
        color: "#000000"
        width:170
        height:170
        x: 425
        y: 216

        ShaderEffectSource {
            id: artistImageSource
            sourceItem: artistImage
            recursive: true
            live: true
        }

        Image {
            source: "qrc:/images/unknown_song.png"
            fillMode: Image.PreserveAspectCrop
            width:170
            height:170
            x: 0
            y: 0
            opacity: 1
            layer.enabled:true
            layer.effect: OpacityMask{
                maskSource: artistImageSource
            }
        }
        Image {
            id: artistImg
            source: ""
            fillMode: Image.PreserveAspectCrop
            width:170
            height:170
            x: 0
            y: 0
            opacity: 1
            layer.enabled:true
            layer.effect: OpacityMask{
                maskSource: artistImageSource
            }
        }
        Text {
            id: playingNow
            text: "Playing Now"
            font.pixelSize: 21
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.weight: Font.Black
            font.family: "Arial-Black"
            color: "#f2f2f2"
            smooth: true
            x: 0
            y: -36
            width: 169
            height: 25
            opacity: 0.70196078431373
            visible: true
        }

        Text {
            id: songTitle
            text: "."
            font.pixelSize: 20
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Arial-Black"
            color: "#ffffff"
            wrapMode: Text.Wrap
            smooth: true
            x: -59
            y: 179
            width: 294
            height: 32
            opacity: 1
        }

        Text {
            id: artistName
            text: "."
            font.pixelSize: 19
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Arial-Black"
            color: "#ffffff"
            smooth: true
            x: 0
            y: 225
            width: 169
            height: 19
            opacity: 0.4
        }
    }

    VideoOutput {
        id: viewfinder
        x: 124
        y: 70
        width: 1236
        height: 698
        anchors.fill: rectangle_1
        anchors.centerIn: parent
        source: camera
        visible: false

    }


    Camera {
        id: camera

    }





    Image {
        id: right_side_blue
        source: "images/right_side_blue.png"
        x: 616
        y: 113
        opacity: 1
        visible: true
    }
    Image {
        id: right_side_red
        source: "images/right_side_red.png"
        x: 616
        y: 113
        opacity: 1
        visible: false
    }
    Image {
        id: right_side_yellow
        source: "images/right_side_yellow.png"
        x: 616
        y: 113
        opacity: 1
        visible: false
    }


    Image {
        id: left_side
        source: "images/left_side.png"
        x: -5
        y: 113
        opacity: 1
    }


    Gauge_animation {
        x: 42
        y: 161
        width: 317
        height: 317
        value: speed_read.text
        maximumValue: 160
        anchors {
            margins: window.height * 0.2
        }
        Text {
            id: min_speed
            x: 84
            y: 264
            color: "#ffffff"
            text: qsTr("0")
            font.pixelSize: 17
            font.family: "GoogleSansDisplay-Bold"
            font.bold: true
        }

        Text {
            id: max_speed
            x: 206
            y: 268
            font.family: "GoogleSansDisplay-Bold"
            color: "#d92a27"
            text: qsTr("160")
            font.pixelSize: 17
            font.bold: true
        }
    }




    Image {
        id: left_side_center
        source: "images/center.png"
        x: 80
        y: 197
        opacity: 1
    }

    Image {
        id: upper_status
        source: "images/upper_status.png"
        clip: true
        x: 0
        y: 0
        opacity: 0.911
    }

    Image {
        id: lower_status
        source: "images/lower_status.png"
        fillMode: Image.PreserveAspectFit
        x: 320
        y: 445
        width: 380
        height: 236
        opacity: 1
    }

    Text {
        id: miles_value
        text: "0000"
        font.pixelSize: 16
        font.family: "Futura"
        color: "#d0d0d0"
        smooth: true
        x: 782
        y: 389
        opacity: 1
    }

    Text {
        id: gear
        text: "Gear"
        font.pixelSize: 20
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        font.capitalization: Font.AllUppercase
        font.family: "Futura"
        font.bold: true
        color: "#e7bf8c"
        smooth: true
        x: 791
        y: 239
        opacity: 1
        visible: true


        Text {
            id: _P
            x: -23
            y: 114
            width: 20
            height: 20
            color: "#ffffff"
            text: qsTr("P")
            font.pixelSize: 25
            horizontalAlignment: Text.AlignHCenter
            font.weight: Font.Black
        }

        Text {
            id: _N
            x: 27
            y: 114
            width: 20
            height: 20
            color: "#7f7f7f"
            text: qsTr("N")
            font.pixelSize: 25
            horizontalAlignment: Text.AlignHCenter
            font.weight: Font.Black
        }

        Text {
            id: _D
            x: 53
            y: 114
            width: 20
            height: 20
            color: "#7f7f7f"
            text: qsTr("D")
            font.pixelSize: 25
            horizontalAlignment: Text.AlignHCenter
            font.weight: Font.Black
        }

        Text {
            id: _R
            x: 0
            y: 114
            width: 20
            height: 20
            color: "#7f7f7f"
            text: qsTr("R")
            font.pixelSize: 25
            horizontalAlignment: Text.AlignHCenter
            font.weight: Font.Black
        }

        Text {
            id: selected_gear
            text: "P"
            font.pixelSize: 80
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "Futura"
            font.bold: true
            color: "#ffffff"
            smooth: true
            x: -60
            y: 22
            width: 170
            height: 86
            opacity: 1
        }

        Text {
            id: drive_mode
            x: -344
            y: -204
            text: "Parking"
            font.pixelSize: 18
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.capitalization: Font.AllUppercase
            font.weight: Font.Black
            font.family: "Futura"
            color: "#e7bf8c"
            smooth: true
            width: 133
            height: 22
            opacity: 1
        }

    }


    Text {
        id: gearAuto
        text: "Gear"
        font.pixelSize: 20
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        font.capitalization: Font.AllUppercase
        font.family: "Futura"
        font.bold: true
        color: "#e7bf8c"
        smooth: true
        x: 791
        y: 239
        opacity: 1
        visible: false

        Text {
            id: auto_selected_gear
            text: "P"
            font.pixelSize: 80
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.family: "GoogleSansDisplay-Bold"
            font.bold: true
            color: "#ffffff"
            smooth: true
            x: -60
            y: 22
            width: 170
            height: 86
            opacity: 1
        }

        Text {
            id: auto_P
            x: -22
            y: 114
            width: 20
            height: 20
            color: "#ffffff"
            text: qsTr("P")
            font.pixelSize: 25
            horizontalAlignment: Text.AlignHCenter
            font.weight: Font.Black
        }

        Text {
            id: auto_N
            x: 4
            y: 114
            width: 20
            height: 20
            color: "#7f7f7f"
            text: qsTr("N")
            font.pixelSize: 25
            horizontalAlignment: Text.AlignHCenter
            font.weight: Font.Black
        }

        Text {
            id: auto_D
            x: 30
            y: 114
            width: 20
            height: 20
            color: "#7f7f7f"
            text: qsTr("D")
            font.pixelSize: 25
            horizontalAlignment: Text.AlignHCenter
            font.weight: Font.Black
        }

        Text {
            id: auto_R
            x: 54
            y: 114
            width: 20
            height: 20
            color: "#7f7f7f"
            text: qsTr("R")
            font.pixelSize: 25
            horizontalAlignment: Text.AlignHCenter
            font.weight: Font.Black
        }

        Text {
            id: auto_drive_mode
            text: "Parking"
            font.pixelSize: 18
            horizontalAlignment: Text.AlignHCenter
            font.capitalization: Font.AllUppercase
            font.weight: Font.Black
            font.family: "Arial"
            color: "#e7bf8c"
            smooth: true
            x: -344
            y: -204
            width: 133
            height: 22
            opacity: 1
        }
    }


    Text {
        id: speed_read
        text: speedValue
        font.pixelSize: 80
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignBottom
        font.family: "Futura"
        font.bold: true
        color: "#ffffff"
        smooth: true
        x: 118
        y: 261
        width: 167
        height: 97
        opacity: 1
        visible: true
        property int speedValue: 0

        SequentialAnimation on speedValue {
            id: speedAnimation
            loops: Animation.Infinite

            NumberAnimation {
                from: 0
                to: 160
                duration: 4000
                easing.type: Easing.InOutQuad
            }

            NumberAnimation {
                from: 160
                to: 0
                duration: 2000
                easing.type: Easing.InOutQuad
            }
        }
    }
    // Start the animation only if hasRealData is false
    Component.onCompleted: {
        if (hasRealData==0) {
            speedAnimation.start()
        }
        else{
            speedAnimation.stop()
        }
    }


    Image {
        id: speed_limit
        source: "images/Set 1/signs.png"
        fillMode: Image.PreserveAspectCrop
        sourceSize.width: 0
        x: 232
        y: 364
        width: 37
        height: 37
        opacity: 1
        visible: true
    }

    Image {
        id: door_open
        source: "images/door_open.png"
        x: 77
        y: 17
        opacity: 1
    }

    Image {
        id: low_bat
        source: "images/low_bat.png"
        x: 782
        y: 72
        opacity: 1
        visible: true
    }

    Image {
        id: parking_break
        source: "images/parking_break.png"
        x: 144
        y: 49
        opacity: 1
        visible: true
    }

    Image {
        id: lower_status1
        x: 785
        y: 444
        width: 380
        height: 236
        opacity: 1
        source: "images/lower_status.png"
        fillMode: Image.PreserveAspectFit
    }


    Image {
        id: seat_belt
        source: "images/seat_belt.png"
        x: 210
        y: 65
        opacity: 1
        visible: true
    }


    Image {
        id: assist_disable
        source: "images/assist_disable.png"
        x: 919
        y: 15
        opacity: 1
        visible: true
    }


    Image {
        id: steering_error
        source: "images/steering_error.png"
        x: 853
        y: 44
        opacity: 1
        visible: true
    }


    Image {
        id: low_beam
        source: "images/low_beam.png"
        x: 521
        y: 551
        width: 41
        opacity: 1
    }



    Image {
        id: high_beam
        source: "images/high_beam.png"
        sourceSize.height: 0
        sourceSize.width: 0
        x: 398
        y: 551
        width: 40
        height: 25
        opacity: 1
    }



    Image {
        id: adaptive_off
        source: "images/adaptive_off.png"
        x: 457
        y: 551
        width: 40
        height: 25
        opacity: 1
    }


    Image {
        id: adaptive_on
        source: "images/adaptive_on.png"
        sourceSize.height: 0
        sourceSize.width: 0
        x: 457
        y: 551
        width: 40
        height: 25
        opacity: 1
    }



    Image {
        id: daytime_light
        source: "images/daytime_light.png"
        x: 579
        y: 553
        width: 46
        opacity: 1
    }


    Text {
        id: alert_
        text: "Alert"
        font.pixelSize: 18
        verticalAlignment: Text.AlignVCenter
        font.capitalization: Font.AllUppercase
        font.weight: Font.Black
        font.family: "Futura"
        color: "#d82927"
        smooth: true
        x: 483
        y: 63
        opacity: 1
        visible: true
    }



    Text {
        id: _cautionMassage
        text: "System Malfunction"
        font.pixelSize: 18
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        font.bold: true
        font.capitalization: Font.AllUppercase
        font.family: "Futura"
        color: "#ffffff"
        smooth: true
        x: 360
        y: 89
        width: 306
        height: 19
        opacity: 1
        visible: true
    }



    Text {
        id: clockdate
        text: "00:00"
        font.pixelSize: 20
        font.family: "Futura"
        font.bold: true
        color: "#ffffff"
        smooth: true
        x: 855
        y: 550
        opacity: 1
    }



    Text {
        id: clockDateSign
        text: "PM"
        font.pixelSize: 12
        font.family: "Futura"
        font.bold: true
        color: "#7f7f7f"
        smooth: true
        x: 919
        y: 556
        opacity: 1
    }



    Text {
        id: tempLabel
        text: "--"
        font.pixelSize: 22
        horizontalAlignment: Text.AlignHCenter
        font.weight: Font.Black
        font.family: "Arial-Black"
        color: "#ffffff"
        smooth: true
        x: 944
        y: 551
        width: 44
        height: 27
        opacity: 1

        Text {
            id: tempDegree
            text: "o"
            font.pixelSize: 15
            font.family: "Futura"
            color: "#7f7f7f"
            smooth: true
            x: 47
            y: -6
            opacity: 1
        }
        Text {
            id: tempSign
            text: "c"
            font.pixelSize: 22
            horizontalAlignment: Text.AlignHCenter
            font.weight: Font.Black
            font.family: "Futura"
            color: "#7f7f7f"
            smooth: true
            x: 50
            y: 0
            width: 22
            height: 27
            opacity: 1
        }
    }




    Image {
        id: rightlabel
        x: 592
        y: 29
        width: 35
        height: 35
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
                duration: 700  // Adjust the duration of the fade-out
            }

            NumberAnimation {
                target: rightlabel
                property: "opacity"
                to: 1
                duration: 500  // Adjust the duration of the fade-in
            }
        }
    }



    Image {
        id: leftlabel
        x: 399
        y: 29
        width: 35
        height: 35
        source: "images/left.png"
        fillMode: Image.PreserveAspectFit
        opacity:0
        visible: true

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
                duration: 700  // Adjust the duration of the fade-out
            }

            NumberAnimation {
                target: leftlabel
                property: "opacity"
                to: 1
                duration: 500  // Adjust the duration of the fade-in
            }
        }
    }





    Connections {
        target: serialManager
        onJsonDataParsed: {
            tempLabel.text = ""+temperature;

            if(door){
                door_open.visible=false;
            }else{
                door_open.visible=true;
            }

            selected_gear.text=gear;
            if (selected_gear.text === "P") {
                drive_mode.text="Parking"
                _P.color = "#ffffff";
            } else {
                _P.color = "#7f7f7f";
            }
            if (selected_gear.text === "N") {
                drive_mode.text="Neutral"
                _N.color = "#ffffff";
            } else {
                _N.color = "#7f7f7f";
            }
            if (selected_gear.text === "D") {
                drive_mode.text="Driving Mode"
                _D.color = "#ffffff";
            } else {
                _D.color = "#7f7f7f";
            }
            if (selected_gear.text === "R") {
                drive_mode.text="Reversing"
                _R.color = "#ffffff";
                camera.start() // Start the camera
                viewfinder.visible = true // Show the viewfinder
            } else {
                _R.color = "#7f7f7f";
                camera.stop() // Stop the camera (not start)
                viewfinder.visible = false // Hide the viewfinder
            }
        }
    }




//    Connections {
//        target: carDataReceiver
//        onSleepStatusChanged: {
//            if(sleepStatus===1){
//                _cautionMassage.visible=true;
//                alert_.visible=true;
//            }
//        }
//    }




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



    Connections{
        target: SpotifyReceiver
        onSpotifiyReceivedData:{
            if(Album_Name===""){
                artistImage.visible=false
                playingNow.visible=false
            }
            else{
                artistImage.visible=true
                playingNow.visible=true
                artistImg.source = Album_Img_URL
                songTitle.text=Track_Name
                artistName.text=Artist_Name
            }

            if(speed_read.text>0&&speed_read.text<60){
                right_side_blue.visible=true
                right_side_yellow.visible=false
                right_side_red.visible=false
            }
            if (speed_read.text>=60&&speed_read.text<120){
                right_side_blue.visible=false
                right_side_yellow.visible=true
                right_side_red.visible=false
            }
            if(speed_read.text>=120){
                right_side_blue.visible=false
                right_side_yellow.visible=false
                right_side_red.visible=true
            }
        }
    }



    Connections{
        target:carDataReceiver
        onCarlaJsonDataParsed:{
            speed_read.text = ""+speed;
            if(leftSignal===1){
                leftlabel.visible=true;
            }
            else{
                leftlabel.visible=false;
            }

            if(rightSignal===1){
                rightlabel.visible=true;
            }
            else{
                rightlabel.visible=false;
            }

            if(sign===1){
                gear.visible=false;
                gearAuto.visible=true;
                auto_selected_gear.text=autoGear;
                if (auto_selected_gear.text === "P") {
                    auto_drive_mode.text="Parking"
                    auto_P.color = "#ffffff";
                } else {
                    auto_P.color = "#7f7f7f";
                }
                if (auto_selected_gear.text === "N") {
                    auto_drive_mode.text="Neutral"
                    auto_N.color = "#ffffff";
                } else {
                    auto_N.color = "#7f7f7f";
                }
                if (auto_selected_gear.text === "D") {
                    auto_drive_mode.text="Driving Mode"
                    auto_D.color = "#ffffff";
                } else {
                    auto_D.color = "#7f7f7f";
                }
                if (auto_selected_gear.text === "R") {
                    auto_drive_mode.text="Reversing"
                    auto_R.color = "#ffffff";
                } else {
                    auto_R.color = "#7f7f7f";
                }
            }
            else{
                gear.visible=true;
                gearAuto.visible=false;
            }
            if(warning===0){
                wanring_massage.visible=false;
                alert_.visible=false;
                _cautionMassage.visible=false;
                daytime_light.visible=false;
                adaptive_on.visible=false;
                adaptive_off.visible=false;
                low_beam.visible=false;
                high_beam.visible=false;
                assist_disable.visible=false;
                steering_error.visible=false;
                speed_limit.visible=false;
                low_bat.visible=false;
                seat_belt.visible=false;
                hasRealData: 1;
            }
            if(handBrake){
                parking_break.visible=true;
            }else{
                parking_break.visible=false;
            }
            if(auto_selected_gear.text === "R"){
                drive_mode.text="Reversing"
                camera.start() // Start the camera
                viewfinder.visible = true // Show the viewfinder
            }else{
                camera.stop() // Stop the camera (not start)
                viewfinder.visible = false // Hide the viewfinder

            }
        }
    }



    Image {
        id: wanring_massage
        x: 327
        y: 335
        source: "images/wanring_massage.png"
        cache: false
        transformOrigin: Item.Center
        width: 366
        height: 190
        opacity: 1
        visible: true
        Item{
            id: engineMulfunction
            visible: true
            Image {
                id: caution_icon_Engine
                x: 36
                y: 29
                width: 42
                height: 38
                source: "images/check-engine-light-icon-1616189100.png"
                sourceSize.height: 0
                sourceSize.width: 0
                fillMode: Image.PreserveAspectFit
            }
            Text {
                id: _errorTitle
                x: 100
                y: 36
                width: 245
                height: 23
                color: "#d92a27"
                text: qsTr("Engine Malfunction")
                font.pixelSize: 19
                verticalAlignment: Text.AlignVCenter
                font.family: "Futura"
                font.capitalization: Font.Capitalize
                font.weight: Font.ExtraBold
            }
            Text {
                id: _errorEngine
                x: 35
                y: 68
                width: 300
                height: 89
                color: "#ffffff"
                text: qsTr("Engine operating at reduced output. Possible to continue. Drive with caution. Have the system checked by nearest service center.")
                font.pixelSize: 17
                verticalAlignment: Text.AlignTop
                lineHeight: 0.9
                wrapMode: Text.Wrap
                font.family: "Futura"
                font.weight: Font.DemiBold
                font.preferShaping: true
                font.kerning: true
            }
        }

        Item{
            id: roadImprefection
            visible: false
            Image {
                id: caution_icon_Road_Imprefection
                x: 32
                y: 13
                width: 60
                height: 60
                source: "images/layer_2.png"
                sourceSize.height: 0
                sourceSize.width: 0
                fillMode: Image.Stretch
            }
            Text {
                id: _errorTitle2
                x: 80
                y: 40
                width: 255
                height: 23
                color: "#d97b27"
                text: qsTr("Road Imprefection Detected")
                font.pixelSize: 18
                verticalAlignment: Text.AlignVCenter
                font.family: "Futura"
                font.capitalization: Font.Capitalize
                font.weight: Font.ExtraBold
            }
            Text {
                id: _errorRoadImprefections
                x: 35
                y: 68
                width: 300
                height: 89
                color: "#ffffff"
                text: qsTr("Alert an imperfection has been dectected please slow down. Drive with caution. Nearest road bump in 10 meters away.")
                font.pixelSize: 17
                verticalAlignment: Text.AlignTop
                lineHeight: 0.9
                wrapMode: Text.Wrap
                font.family: "Futura"
                font.weight: Font.DemiBold
                font.preferShaping: true
                font.kerning: true
            }
        }

        Item{
            id: radarDetected
            visible: false
            Image {
                id: caution_icon_errorRadarDetected
                x: 36
                y: 29
                width: 42
                height: 38
                source: "images/layer_2.png"
                sourceSize.height: 0
                sourceSize.width: 0
                fillMode: Image.PreserveAspectFit
            }
            Text {
                id: _errorTitle3
                x: 100
                y: 36
                width: 245
                height: 23
                color: "#d92a27"
                text: qsTr("Speed Radar Detected")
                font.pixelSize: 19
                verticalAlignment: Text.AlignVCenter
                font.family: "Futura"
                font.capitalization: Font.Capitalize
                font.weight: Font.ExtraBold
            }
            Text {
                id: _errorRadarDetected
                x: 35
                y: 68
                width: 312
                height: 89
                color: "#e8c291"
                text: qsTr("Attention, Radar is near you, Please slow down and keep focused on the road.")
                font.pixelSize: 17
                verticalAlignment: Text.AlignTop
                lineHeight: 0.9
                wrapMode: Text.Wrap
                font.family: "Futura"
                font.weight: Font.DemiBold
                font.preferShaping: true
                font.kerning: true
            }
        }

        Item{
            id: airBagError
            visible: false
            Image {
                id: caution_icon_airBagError
                x: 39
                y: 29
                width: 39
                height: 35
                source: "images/367-200.png"
                sourceSize.height: 0
                sourceSize.width: 0
                fillMode: Image.Stretch
            }
            Text {
                id: _errorTitle4
                x: 84
                y: 35
                width: 254
                height: 23
                color: "#d92a27"
                text: qsTr("Airbag System Fault")
                font.pixelSize: 19
                verticalAlignment: Text.AlignVCenter
                font.family: "Futura"
                font.capitalization: Font.Capitalize
                font.weight: Font.ExtraBold
            }
            Text {
                id: _errorAirbag
                x: 35
                y: 67
                width: 303
                height: 95
                color: "#ffffff"
                text: qsTr("SRS system detected a fault in the system. Continue wearing safety belt. consult nearest service center.")
                font.pixelSize: 17
                verticalAlignment: Text.AlignTop
                lineHeight: 0.9
                wrapMode: Text.Wrap
                font.family: "Futura"
                font.weight: Font.DemiBold
                font.preferShaping: true
                font.kerning: true
            }
        }


        Item{
            id: electricSteeringError
            visible: false
            Image {
                id: caution_icon_electricSteeringError
                x: 35
                y: 32
                width: 36
                height: 33
                source: "images/steering_error.png"
                sourceSize.height: 0
                sourceSize.width: 0
                fillMode: Image.PreserveAspectFit
            }
            Text {
                id: _errorTitle5
                x: 82
                y: 37
                width: 238
                height: 23
                color:"#d92a27"
                text: qsTr("Electric Steering Failed")
                font.pixelSize: 19
                verticalAlignment: Text.AlignVCenter
                font.family: "Futura"
                font.capitalization: Font.Capitalize
                font.weight: Font.ExtraBold
            }
            Text {
                id: _errorSteering
                x: 35
                y: 65
                width: 312
                height: 89
                color: "#ffffff"
                text: qsTr("Steering system may have some communictaion issues or broken parts. Drive with caution. Have the system checked by nearest service center.")
                font.pixelSize: 17
                verticalAlignment: Text.AlignTop
                lineHeight: 0.9
                wrapMode: Text.Wrap
                font.family: "Futura"
                font.weight: Font.DemiBold
                font.preferShaping: true
                font.kerning: true
            }
        }





    }

    Text {
        id: miles
        x: 824
        y: 389
        opacity: 1
        color: "#7f7f7f"
        text: "mi"
        font.pixelSize: 16
        smooth: true
        font.family: "Futura"
    }

}
