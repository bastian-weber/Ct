#include "ImportSettingsDialog.h"

namespace ct {

	ImportSettingsDialog::ImportSettingsDialog(std::shared_ptr<QSettings> settings, QWidget* parent)
		: settings(settings),
		QDialog(parent) {
		this->setWindowModality(Qt::WindowModal);
		this->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
		this->setWindowFlags(this->windowFlags() & ~Qt::WindowContextHelpButtonHint);

		this->setWindowTitle(tr("Volume Parameters"));

		this->okButton = new QPushButton(tr("&Ok"), this);
		this->okButton->setDefault(true);
		QObject::connect(this->okButton, SIGNAL(clicked()), this, SLOT(accept()));
		this->cancelButton = new QPushButton(tr("&Cancel"), this);
		QObject::connect(this->cancelButton, SIGNAL(clicked()), this, SLOT(reject()));

		this->xSpinBox = new QSpinBox;
		this->xSpinBox->setRange(1, std::numeric_limits<int>::max());
		this->xSpinBox->setSingleStep(1);
		this->ySpinBox = new QSpinBox;
		this->ySpinBox->setRange(1, std::numeric_limits<int>::max());
		this->ySpinBox->setSingleStep(1);
		this->zSpinBox = new QSpinBox;
		this->zSpinBox->setRange(1, std::numeric_limits<int>::max());
		this->zSpinBox->setSingleStep(1);
		QObject::connect(this->xSpinBox, SIGNAL(valueChanged(int)), this, SLOT(updateSize()));
		QObject::connect(this->ySpinBox, SIGNAL(valueChanged(int)), this, SLOT(updateSize()));
		QObject::connect(this->zSpinBox, SIGNAL(valueChanged(int)), this, SLOT(updateSize()));
		this->littleEndianRadioButton = new QRadioButton(tr("Little endian"), this);
		this->bigEndianRadioButton = new QRadioButton(tr("Big endian"), this);
		this->zFastestRadioButton = new QRadioButton(tr("Z fastest"), this);
		this->xFastestRadioButton = new QRadioButton(tr("X fastest"), this);
		this->byteOrderGroup = new QButtonGroup(this);
		this->indexOrderGroup = new QButtonGroup(this);
		this->byteOrderGroup->addButton(littleEndianRadioButton);
		this->byteOrderGroup->addButton(bigEndianRadioButton);
		this->indexOrderGroup->addButton(zFastestRadioButton);
		this->indexOrderGroup->addButton(xFastestRadioButton);
		this->littleEndianRadioButton->setChecked(true);
		this->xFastestRadioButton->setChecked(true);
		this->requiredSizeLabel = new QLabel("");
		this->actualSizeLabel = new QLabel("");

		this->dataTypeComboBox = new QComboBox(this);
		this->dataTypeComboBox->insertItem(0, "32bit float");
		this->dataTypeComboBox->insertItem(1, "64bit double");
		this->dataTypeComboBox->insertItem(2, "8bit signed integer");
		this->dataTypeComboBox->insertItem(3, "8bit unsigned integer");
		this->dataTypeComboBox->insertItem(4, "16bit signed integer");
		this->dataTypeComboBox->insertItem(5, "16bit unsigned integer");
		this->dataTypeComboBox->insertItem(6, "32bit signed integer");
		this->dataTypeComboBox->insertItem(7, "32bit unsigned integer");
		QObject::connect(this->dataTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(updateSize()));

		this->mainLayout = new QVBoxLayout(this);

		this->formLayout = new QFormLayout();
		this->formLayout->setSpacing(10);
		this->formLayout->addRow(tr("X size:"), this->xSpinBox);
		this->formLayout->addRow(tr("Y size:"), this->ySpinBox);
		this->formLayout->addRow(tr("Z size:"), this->zSpinBox);
		this->formLayout->addRow(tr("Byte order:"), this->littleEndianRadioButton);
		this->formLayout->addRow("", this->bigEndianRadioButton);
		this->formLayout->addRow(tr("Index order:"), this->xFastestRadioButton);
		this->formLayout->addRow("", this->zFastestRadioButton);
		this->formLayout->addRow("Data type:", this->dataTypeComboBox);
		this->formLayout->addRow(tr("Actual filesize:"), this->requiredSizeLabel);
		this->formLayout->addRow(tr("Resulting filesize:"), this->actualSizeLabel);

		this->buttonLayout = new QHBoxLayout();
		this->buttonLayout->addStretch(1);
		this->buttonLayout->addWidget(okButton);
		this->buttonLayout->addWidget(cancelButton);

		this->mainLayout->addLayout(this->formLayout);
		this->mainLayout->addSpacing(10);
		this->mainLayout->addLayout(this->buttonLayout);

		this->setLayout(this->mainLayout);
		this->layout()->setSizeConstraint(QLayout::SetFixedSize);
	}

	int ImportSettingsDialog::execForFilesize(size_t requiredSize) {
		this->requiredSize = requiredSize;
		this->requiredSizeLabel->setText(QString::number(requiredSize).append(" bytes"));
		return this->exec();
	}

	size_t ImportSettingsDialog::getXSize() const {
		return static_cast<size_t>(this->xSpinBox->value());
	}

	size_t ImportSettingsDialog::getYSize() const {
		return static_cast<size_t>(this->ySpinBox->value());
	}

	size_t ImportSettingsDialog::getZSize() const {
		return static_cast<size_t>(this->zSpinBox->value());
	}

	IndexOrder ImportSettingsDialog::getIndexOrder() const {
		if (this->zFastestRadioButton->isChecked()) {
			return IndexOrder::Z_FASTEST;
		}
		return IndexOrder::X_FASTEST;
	}

	QDataStream::ByteOrder ImportSettingsDialog::getByteOrder() const {
		if (this->littleEndianRadioButton->isChecked()) {
			return QDataStream::LittleEndian;
		}
		return QDataStream::BigEndian;
	}

	DataType ImportSettingsDialog::getDataType() const {
		int index = this->dataTypeComboBox->currentIndex();
		if (index == 0) {
			return DataType::FLOAT32;
		} else if (index == 1) {
			return DataType::DOUBLE64;
		} else if (index == 2) {
			return DataType::INT8;
		} else if (index == 3) {
			return DataType::UINT8;
		} else if (index == 4) {
			return DataType::INT16;
		} else if (index == 5) {
			return DataType::UINT16;
		} else if (index == 6) {
			return DataType::INT32;
		} else if (index == 7) {
			return DataType::UINT32;
		}
		return DataType::FLOAT32;
	}

	void ImportSettingsDialog::setXSize(size_t xSize) {
		this->xSpinBox->setValue(static_cast<int>(xSize));
	}

	void ImportSettingsDialog::setYSize(size_t ySize) {
		this->ySpinBox->setValue(static_cast<int>(ySize));
	}

	void ImportSettingsDialog::setZSize(size_t zSize) {
		this->zSpinBox->setValue(static_cast<int>(zSize));
	}

	void ImportSettingsDialog::setIndexOrder(IndexOrder indexOrder) {
		if (indexOrder == IndexOrder::Z_FASTEST) {
			this->zFastestRadioButton->setChecked(true);
		} else {
			this->xFastestRadioButton->setChecked(true);
		}
	}

	void ImportSettingsDialog::setByteOrder(QDataStream::ByteOrder byteOrder) {
		if (byteOrder == QDataStream::LittleEndian) {
			this->littleEndianRadioButton->setChecked(true);
		} else {
			this->bigEndianRadioButton->setChecked(true);
		}
	}

	void ImportSettingsDialog::showEvent(QShowEvent * e) {
		if (this->parentWidget() != 0) {
			move(this->parentWidget()->window()->frameGeometry().topLeft() + this->parentWidget()->window()->rect().center() - this->rect().center());
		}
		this->updateSize();
	}
	
	//============================================================================== PROTECTED ==============================================================================\\


	//=============================================================================== PRIVATE ===============================================================================\\


	//============================================================================ PRIVATE SLOTS =============================================================================\\

	void ImportSettingsDialog::updateSize() {
		size_t voxelSize = 4;
		DataType type = this->getDataType();
		if (type == DataType::FLOAT32) {
			voxelSize = 4;
		} else if (type == DataType::DOUBLE64) {
			voxelSize = 8;
		} else if (type == DataType::UINT8) {
			voxelSize = sizeof(uint8_t);
		} else if (type == DataType::INT8) {
			voxelSize = sizeof(int8_t);
		} else if (type == DataType::UINT16) {
			voxelSize = sizeof(uint16_t);
		} else if (type == DataType::INT16) {
			voxelSize = sizeof(int16_t);
		} else if (type == DataType::UINT32) {
			voxelSize = sizeof(uint32_t);
		} else if (type == DataType::INT32) {
			voxelSize = sizeof(int32_t);
		}

		size_t currentSize = voxelSize * static_cast<size_t>(this->xSpinBox->value())*static_cast<size_t>(this->ySpinBox->value())*static_cast<size_t>(this->zSpinBox->value());
		this->actualSizeLabel->setText(QString::number(currentSize).append(" bytes"));
		if (currentSize == this->requiredSize) {
			this->actualSizeLabel->setStyleSheet("QLabel { }");
			this->okButton->setEnabled(true);
		} else {
			this->actualSizeLabel->setStyleSheet("QLabel { color: red; }");
			this->okButton->setEnabled(false);
		}
	}

}