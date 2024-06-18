// // can_bus.cpp

// #include "can_bus.h"
// #include <QDebug>

// CanReader::CanReader(QObject *parent) : QObject(parent) {
//     // Connect to CAN bus
//     m_canDevice = QCanBus::instance()->createDevice("socketcan", "can0");
//     if (!m_canDevice || !m_canDevice->connectDevice()) {
//         qDebug() << "Error: Failed to connect to CAN bus";
//         return;
//     }

//     // Set up message filters
//     QList<QCanBusFrame> filterList;
//     filterList.append(QCanBusFrame(0x555, QByteArray(), QCanBusFrame::StandardFrame));
//     filterList.append(QCanBusFrame(0x123, QByteArray(), QCanBusFrame::StandardFrame));
//     m_canDevice->setConfigurationParameter(QCanBusDevice::RawFilterKey, QVariant::fromValue(filterList));

//     // Connect to UDP socket
//     m_udpSocket = new QUdpSocket(this);
//     m_udpSocket->connectToHost("10.42.0.22", 12345);
// }

// void CanReader::readMessages() {
//     while (m_canDevice->framesAvailable()) {
//         QCanBusFrame frame = m_canDevice->readFrame();
//         if (frame.frameId() == 0x555) {
//             QByteArray data = frame.payload();
//             int throttle = static_cast<int>(data.at(1));
//             int brake = static_cast<int>(data.at(0));
//             sendJsonData(throttle, brake, m_lastSteering);
//         } else if (frame.frameId() == 0x123) {
//             QByteArray data = frame.payload();
//             float steering = *reinterpret_cast<float*>(data.data());
//             m_lastSteering = steering;
//             sendJsonData(m_lastThrottle, m_lastBrake, steering);
//         }
//     }
// }

// void CanReader::sendJsonData(int throttle, int brake, float steering) {
//     QJsonObject dataObject;
//     dataObject["throttle"] = throttle;
//     dataObject["brake"] = brake;
//     dataObject["steering"] = steering;

//     QJsonDocument jsonDocument(dataObject);
//     QByteArray jsonData = jsonDocument.toJson(QJsonDocument::Compact);
//     m_udpSocket->writeDatagram(jsonData, QHostAddress("10.42.0.22"), 12345);
// }
