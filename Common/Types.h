//note: this class is header-only

#ifndef CT_COMPLETIONSTATUS
#define CT_COMPLETIONSTATUS

//Qt
#include <QtCore/QtCore>

namespace ct {

	struct CompletionStatus {
		CompletionStatus() : successful(true), userInterrupted(false) { };
		CompletionStatus(bool successful, bool userInterrupted, QString const& errorMessage = QString()) : successful(successful), userInterrupted(userInterrupted), errorMessage(errorMessage) { }
		static CompletionStatus success() { return CompletionStatus(true, false); }
		static CompletionStatus interrupted() { return CompletionStatus(false, true); }
		static CompletionStatus error(QString const& errorMessage) { return CompletionStatus(false, false, errorMessage); }
		bool successful;
		bool userInterrupted;
		QString errorMessage;
	};

	enum class ImageBitDepth {
		CHANNEL_8_BIT,
		CHANNEL_16_BIT
	};

	enum class DataType {
		FLOAT32,
		INT16
	};

}

#endif