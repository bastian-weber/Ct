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
		QObject::connect(this, SIGNAL(accepted()), this, SLOT(saveSettings()));
		this->cancelButton = new QPushButton(tr("&Cancel"), this);
		QObject::connect(this->cancelButton, SIGNAL(clicked()), this, SLOT(reject()));

		this->xSpinBox = new QSpinBox(this);
		this->xSpinBox->setRange(1, std::numeric_limits<int>::max());
		this->xSpinBox->setSingleStep(1);
		this->ySpinBox = new QSpinBox(this);
		this->ySpinBox->setRange(1, std::numeric_limits<int>::max());
		this->ySpinBox->setSingleStep(1);
		this->zSpinBox = new QSpinBox(this);
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
		this->dataTypeComboBox->insertItem(0, tr("32 bit float"));
		this->dataTypeComboBox->insertItem(1, tr("64 bit double"));
		this->dataTypeComboBox->insertItem(2, tr("8 bit unsigned integer"));
		this->dataTypeComboBox->insertItem(3, tr("8 bit signed integer"));
		this->dataTypeComboBox->insertItem(4, tr("16 bit unsigned integer"));
		this->dataTypeComboBox->insertItem(5, tr("16 bit signed integer"));
		this->dataTypeComboBox->insertItem(6, tr("32 bit unsigned integer"));
		this->dataTypeComboBox->insertItem(7, tr("32 bit signed integer"));
		QObject::connect(this->dataTypeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(updateSize()));

		headerSpinBox = new QSpinBox;
		this->headerSpinBox->setRange(0, std::numeric_limits<int>::max());
		headerSpinBox->setSuffix(" bytes");
		this->headerSpinBox->setSingleStep(1);
		QObject::connect(this->headerSpinBox, SIGNAL(valueChanged(int)), this, SLOT(updateSize()));
		mirrorXCheckbox = new QCheckBox(tr("Mirror x-axis"), this);
		mirrorYCheckbox = new QCheckBox(tr("Mirror y-axis"), this);
		mirrorZCheckbox = new QCheckBox(tr("Mirror z-axis"), this);

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
		this->formLayout->addRow(tr("Data type:"), this->dataTypeComboBox);
		this->formLayout->addRow(tr("Header offset:"), this->headerSpinBox);
		this->formLayout->addRow(tr("Axes orientation:"), this->mirrorXCheckbox);
		this->formLayout->addRow("", this->mirrorYCheckbox);
		this->formLayout->addRow("", this->mirrorZCheckbox);
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

		this->setDefaultValues();
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
		return static_cast<DataType>(this->dataTypeComboBox->currentIndex());
	}

	size_t ImportSettingsDialog::getHeaderOffset() const {
		return this->headerSpinBox->value();
	}

	bool ImportSettingsDialog::getMirrorX() const {
		return this->mirrorXCheckbox->isChecked();
	}

	bool ImportSettingsDialog::getMirrorY() const {
		return this->mirrorYCheckbox->isChecked();
	}

	bool ImportSettingsDialog::getMirrorZ() const {
		return this->mirrorZCheckbox->isChecked();
	}

	void ImportSettingsDialog::setXSize(int xSize) {
		this->xSpinBox->setValue(xSize);
	}

	void ImportSettingsDialog::setYSize(int ySize) {
		this->ySpinBox->setValue(ySize);
	}

	void ImportSettingsDialog::setZSize(int zSize) {
		this->zSpinBox->setValue(zSize);
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

	void ImportSettingsDialog::setDataType(DataType dataType) {
		this->dataTypeComboBox->setCurrentIndex(static_cast<int>(dataType));
	}

	void ImportSettingsDialog::setHeaderOffset(int size) {
		this->headerSpinBox->setValue(size);
	}

	void ImportSettingsDialog::setMirrorX(bool value) {
		this->mirrorXCheckbox->setChecked(value);
	}

	void ImportSettingsDialog::setMirrorY(bool value) {
		this->mirrorYCheckbox->setChecked(value);
	}

	void ImportSettingsDialog::setMirrorZ(bool value) {
		this->mirrorZCheckbox->setChecked(value);
	}

	void ImportSettingsDialog::showEvent(QShowEvent * e) {
		if (this->parentWidget() != 0) {
			move(this->parentWidget()->window()->frameGeometry().topLeft() + this->parentWidget()->window()->rect().center() - this->rect().center());
		}
		this->updateSize();
	}

	void ImportSettingsDialog::setDefaultValues() {
		this->settings->beginGroup("Import");
		this->xSpinBox->setValue(this->settings->value("xSize", 1).toInt());
		this->ySpinBox->setValue(this->settings->value("ySize", 1).toInt());
		this->zSpinBox->setValue(this->settings->value("zSize", 1).toInt());
		if (this->settings->value("byteOrder", "littleEndian").toString() == "littleEndian") {
			this->littleEndianRadioButton->setChecked(true);
		} else {
			this->bigEndianRadioButton->setChecked(true);
		}
		if (this->settings->value("indexOrder", "xFastest").toString() == "xFastest") {
			this->xFastestRadioButton->setChecked(true);
		} else {
			this->zFastestRadioButton->setChecked(true);
		}
		this->dataTypeComboBox->setCurrentIndex(this->settings->value("dataType", 0).toInt());
		if (this->settings->value("mirrorX", true).toBool()) this->mirrorXCheckbox->setChecked(true);
		if (this->settings->value("mirrorY", true).toBool()) this->mirrorYCheckbox->setChecked(true);
		if (this->settings->value("mirrorZ", true).toBool()) this->mirrorZCheckbox->setChecked(true);
		this->settings->endGroup();
	}

	void ImportSettingsDialog::saveSettings() {
		this->settings->beginGroup("Import");
		this->settings->setValue("xSize", this->xSpinBox->value());
		this->settings->setValue("ySize", this->ySpinBox->value());
		this->settings->setValue("zSize", this->zSpinBox->value());
		if (this->littleEndianRadioButton->isChecked()) {
			this->settings->setValue("byteOrder", "littleEndian");
		} else {
			this->settings->setValue("byteOrder", "bigEndian");
		}
		if (this->xFastestRadioButton->isChecked()) {
			this->settings->setValue("indexOrder", "xFastest");
		} else {
			this->settings->setValue("indexOrder", "zFastest");
		}
		this->settings->setValue("dataType", this->dataTypeComboBox->currentIndex());
		this->settings->setValue("mirrorX", this->mirrorXCheckbox->isChecked());
		this->settings->setValue("mirrorY", this->mirrorYCheckbox->isChecked());
		this->settings->setValue("mirrorZ", this->mirrorZCheckbox->isChecked());
		this->settings->endGroup();
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

		size_t currentSize = voxelSize * static_cast<size_t>(this->xSpinBox->value())*static_cast<size_t>(this->ySpinBox->value())*static_cast<size_t>(this->zSpinBox->value()) + this->headerSpinBox->value();
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