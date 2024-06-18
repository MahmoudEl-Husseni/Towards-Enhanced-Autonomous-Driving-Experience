#include "cardatareceiver.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>

CarDataReceiver::CarDataReceiver(QObject *parent) : QObject(parent) {
    udpSocket.bind(QHostAddress::Any, 12345); // Bind to the same port used by the Python sender

    connect(&udpSocket, &QUdpSocket::readyRead, this, &CarDataReceiver::processPendingDatagrams);
}

void CarDataReceiver::processPendingDatagrams() {
    while (udpSocket.hasPendingDatagrams()) {
        QByteArray datagram;
        datagram.resize(udpSocket.pendingDatagramSize());
        QHostAddress sender;
        quint16 senderPort;

        udpSocket.readDatagram(datagram.data(), datagram.size(), &sender, &senderPort);

        // Decode the received data as a JSON object
        QJsonParseError jsonError;
        QJsonDocument jsonDoc = QJsonDocument::fromJson(datagram, &jsonError);
        if (jsonError.error != QJsonParseError::NoError) {
            qDebug() << "Error decoding JSON data:" << jsonError.errorString();
            continue;
        }

        if (jsonDoc.isObject()) {
            QJsonObject jsonObj = jsonDoc.object();
            // Extract data from the JSON object
            int speed = jsonObj["speed"].toDouble();
            QString alart = jsonObj["alart"].toString();
            int sign = jsonObj["autoPilot"].toInt();
            QString autoGear= jsonObj["autoGear"].toString();
            int leftSignal=jsonObj["leftBlink"].toInt();
            int rightSignal=jsonObj["rightBlink"].toInt();
            int warning=jsonObj["warning"].toInt();
            int handBrake=jsonObj["handBrake"].toInt();
            float brake=jsonObj["brake"].toDouble();
            emit carlaJsonDataParsed(speed, alart, sign, autoGear,leftSignal,rightSignal,warning,handBrake,brake);
//            qDebug() << "Sign:" << sign;
//            qDebug() << "gear:" << autoGear;
//            qDebug() << "LB:" << leftSignal;
        }
    }
}
