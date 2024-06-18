#include "SpotifyReceiver.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QDebug>


SpotifyReceiver::SpotifyReceiver(QObject *parent) : QObject(parent)
{
    udpSocket.bind(QHostAddress::Any, 12356); // Adjust the port as needed

    connect(&udpSocket, &QUdpSocket::readyRead, this, &SpotifyReceiver::processPendingDatagrams);
}

void SpotifyReceiver::processPendingDatagrams() {
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
            QString Track_Name=jsonObj["trackName"].toString();
            QString Artist_Name=jsonObj["artistName"].toString();
            QString Album_Name=jsonObj["albumName"].toString();
            QString Album_Img_URL=jsonObj["albumURL"].toString();
            emit spotifiyReceivedData(Track_Name, Artist_Name, Album_Name, Album_Img_URL);
            // qDebug()<<Artist_Name;

        }
    }
}
