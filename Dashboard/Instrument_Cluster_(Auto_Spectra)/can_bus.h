// // can_bus.h

// #ifndef CAN_BUS_H
// #define CAN_BUS_H

// #include <QObject>
// #include <QUdpSocket>
// #include <QCanBus>
// #include <QCanBusDevice>
// #include <QCanBusFrame>
// #include <QJsonDocument>
// #include <QJsonObject>
// #include <QJsonArray>

// class CanReader : public QObject {
//     Q_OBJECT
// public:
//     explicit CanReader(QObject *parent = nullptr);
// public slots:
//     void readMessages();
// private:
//     void sendJsonData(int throttle, int brake, float steering);
// private:
//     QCanBusDevice *m_canDevice;
//     QUdpSocket *m_udpSocket;
//     float m_lastSteering;
//     int m_lastThrottle;
//     int m_lastBrake;
// };

// #endif // CAN_BUS_H
