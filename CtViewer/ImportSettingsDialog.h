#ifndef CT_IMPORTSETTINGSDIALOG
#define CT_IMPORTSETTINGSDIALOG

#include <iostream>
#include <memory>
#include <vector>

//Qt
#include <QtCore/QtCore>
#include <QtGui/QtGui>
#include <QtWidgets/QtWidgets>

#include "Volume.h"
#include "Types.h"

namespace ct {

	//forward declaration
	enum class DataType;

	class ImportSettingsDialog : public QDialog {
		Q_OBJECT
	public:
		ImportSettingsDialog(std::shared_ptr<QSettings> settings, QWidget* parent = 0);
		int execForFilesize(size_t requiredSize);
		size_t getXSize() const;
		size_t getYSize() const;
		size_t getZSize() const;
		IndexOrder getIndexOrder() const;
		QDataStream::ByteOrder getByteOrder() const;
		DataType getDataType() const;
		size_t getHeaderOffset() const;
		bool getMirrorX() const;
		bool getMirrorY() const;
		bool getMirrorZ() const;
		void setXSize(int xSize);
		void setYSize(int ySize);
		void setZSize(int zSize);
		void setIndexOrder(IndexOrder indexOrder);
		void setByteOrder(QDataStream::ByteOrder byteOrder);
		void setDataType(DataType dataType);
		void setHeaderOffset(int size);
		void setMirrorX(bool value);
		void setMirrorY(bool value);
		void setMirrorZ(bool value);
	protected:
		void showEvent(QShowEvent* e);
	private:
		//functions

		//variables
		std::shared_ptr<QSettings> settings;
		size_t requiredSize = 0;
		//widgets
		QVBoxLayout* mainLayout;
		QFormLayout* formLayout;
		QHBoxLayout* buttonLayout;
		QPushButton* okButton;
		QPushButton* cancelButton;
		QSpinBox* xSpinBox;
		QSpinBox* ySpinBox;
		QSpinBox* zSpinBox;
		QRadioButton* littleEndianRadioButton;
		QRadioButton* bigEndianRadioButton;
		QRadioButton* xFastestRadioButton;
		QRadioButton* zFastestRadioButton;
		QButtonGroup* byteOrderGroup;
		QButtonGroup* indexOrderGroup;
		QComboBox* dataTypeComboBox;
		QSpinBox* headerSpinBox;
		QCheckBox* mirrorXCheckbox;
		QCheckBox* mirrorYCheckbox;
		QCheckBox* mirrorZCheckbox;
		QLabel* actualSizeLabel;
		QLabel* requiredSizeLabel;
	private slots:
		void updateSize();
	};
}
#endif