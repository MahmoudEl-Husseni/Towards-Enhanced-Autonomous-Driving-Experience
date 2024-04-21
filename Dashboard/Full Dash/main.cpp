#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include "serialmanager.h"

int main(int argc, char *argv[])
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
        QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);

    // Register the SerialManager type
    qmlRegisterType<SerialManager>("com.company.serialmanager", 1, 0, "SerialManager" );

    // Create a single instance of SerialManager
    SerialManager serialManager;

    QQmlApplicationEngine engine;

    // Set SerialManager as a context property for the first engine
    engine.rootContext()->setContextProperty("serialManager", &serialManager);

    const QUrl url(QStringLiteral("qrc:/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl){
            QCoreApplication::exit(-1);
        }
    }, Qt::QueuedConnection);

    // Load the main.qml file
    engine.load(url);
    if (engine.rootObjects().isEmpty())
        return -1;

    // Use the same SerialManager instance for the second engine
    QQmlApplicationEngine engine2;
    engine2.rootContext()->setContextProperty("serialManager", &serialManager);

    const QUrl url2(QStringLiteral("qrc:/main2.qml"));
    QObject::connect(&engine2, &QQmlApplicationEngine::objectCreated,
                      &app, [url2](QObject *obj, const QUrl &objUrl) {
         if (!obj && url2 == objUrl){
             QCoreApplication::exit(-1);
         }
     }, Qt::QueuedConnection);

    // Load the main2.qml file
    engine2.load(url2);
    if (engine2.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
