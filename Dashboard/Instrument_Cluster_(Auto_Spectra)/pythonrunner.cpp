// pythonrunner.cpp
#include "pythonrunner.h"
#include <QFileInfo>
#include <QDebug>
#include <QCoreApplication>

PythonRunner::PythonRunner(QObject *parent) : QObject(parent) {
    connect(&process, &QIODevice::readyRead, this, &PythonRunner::readStandardOutput);
    connect(&process, &QIODevice::readyRead, this, &PythonRunner::readStandardError);
    connect(&process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &PythonRunner::processFinished);
}

void PythonRunner::runPythonScript() {
    QString appDir = QCoreApplication::applicationDirPath();
    // Construct the relative path to the Python script
    QString scriptRelativePath = "../../Spotify.py";
    QString scriptPath = appDir + "/" + scriptRelativePath;

    QFileInfo fileInfo(scriptPath);
    QString absoluteScriptPath = fileInfo.absoluteFilePath();

    process.setWorkingDirectory(fileInfo.absolutePath());
    process.start("python", QStringList() << absoluteScriptPath);


    if (!process.waitForStarted()) {
        qDebug() << "Failed to start the script process:" << process.errorString();
        return;
    }
}

void PythonRunner::readStandardOutput() {
    QString output = process.readAllStandardOutput().simplified();
    qDebug() << "Script output:" << output;
    emit pythonScriptOutput(output);
}

void PythonRunner::readStandardError() {
    QString errorOutput = process.readAllStandardError().simplified();
    qDebug() << "Script produced an error:" << errorOutput;
    emit pythonScriptError(errorOutput);
}

void PythonRunner::processFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    if (exitCode != 0) {
        qDebug() << "Script exited with an error:" << exitCode;
    }

    qDebug() << "Script execution finished with exit code:" << exitCode;
}
