import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Controls 2.12
import QtMultimedia 5.12
import QtQuick.Controls.Styles 1.4
import QtGraphicalEffects 1.0
import com.company.serialmanager 1.0
import MyPythonScript 1.0
import com.company.cardatareceiver 1.0
import spotifyreceiver 1.0

Window {
    id: window
    height: 600
    visible: true
    width: 1024
    color: "#3e3c3c"
    maximumHeight: 600
    maximumWidth: 1024
    minimumHeight: 600
    minimumWidth: 1024
    title: qsTr("Instrument Cluster")
     // visibility: "FullScreen"
//     screen: Qt.application.primaryScreen
//    property int autoPilot: 1
    Instrument_Cluster{

    }

}








