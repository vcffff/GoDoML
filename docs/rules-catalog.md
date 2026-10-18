# Extracted video-rule catalog

Source: `data/forte_kz_regulatory_rules.json`. This document records the supplied JSON; it does not add laws or establish legal conclusions from video.

Source SHA-256: `fb72ce6b581eb919a3321e23cac1c4f72aa3679500099d00e6bf61e78dc399b1`. The source records its normative check date as `2026-10-17`.

The complete JSON contains 3 documents, 156 normative sections, and 31 video rules referring to 52 distinct sections. All original rule IDs, event descriptions, observations, conditions, exceptions, and limitations are reproduced below. These are analytical video checks, not quotations of the law.

20 rules allow visual candidates when all stated conditions, exclusions, visibility, and sequence requirements are established. 11 require external confirmation and cannot become confirmed violations from video alone. A source mode is a feasibility constraint, not measured model accuracy.

For visual candidates, observable evidence may include transfers, counting actions, leaving a workstation, packaging, seals, and visible room equipment. Ownership, employee authority, transaction context, hidden locations, digital documents, and internal policies must remain unknown unless established. An isolated frame cannot establish that an action never occurred. Sampling a whole video is not equivalent to continuous visibility of the relevant operation.

The loader `cashml.rules.load_rules` returns rules keyed by original IDs. Legacy API fields `type_id` and `event_title` alias the original ID and event description; `condition_0`, `condition_1`, etc. preserve condition order. `source_rule` retains the unchanged original object. `legal_basis_sections` resolves every reference to the complete original section; document metadata, exceptions, evidence requirements, interpretation constraints, and applicable transitional provisions remain available. Duplicate IDs, broken references, missing required fields, and malformed catalogs fail closed.

The source JSON is explicitly allowlisted in Git and Docker; videos, labels, and results remain excluded. The default configuration loads this complete file. `rules.json` is retained only for legacy replay and does not contain the complete regulatory catalog.

## Source documents

- `NBK_231`: [Об утверждении Правил осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей в банках второго уровня, филиалах банков-нерезидентов Республики Казахстан, Национальном операторе почты и юридических лицах, исключительной деятельностью которых является инкассация банкнот, монет и ценностей](https://adilet.zan.kz/rus/docs/V1900019680); act 231 dated 2019-11-29, recorded revision 2026-07-12.
  Scope (original): Применимость к конкретному виду операции, помещению и роли должна быть установлена до сопоставления с видео.
- `NBK_14`: [Об утверждении Правил организации охраны и устройства помещений банков второго уровня, филиалов банков-нерезидентов Республики Казахстан, Национального оператора почты, юридических лиц, исключительной деятельностью которых является инкассация банкнот, монет и ценностей, и юридических лиц, осуществляющих деятельность исключительно через обменные пункты на основании лицензии Национального Банка Республики Казахстан на обменные операции с наличной иностранной валютой](https://adilet.zan.kz/rus/docs/V2000020075); act 14 dated 2020-02-24, recorded revision 2024-12-24.
  Scope (original): Разделы 1–3 Правил для банков и инкассаторских организаций; глава 4 об уполномоченных обменных организациях не включена как отдельная сфера применения. Исключения пунктов 9, 10 и 18 сохранены.
- `ARRFR_21`: [Об утверждении Инструкции по организации антитеррористической защиты расчетно-кассовых отделений банков второго уровня, уязвимых в террористическом отношении](https://adilet.zan.kz/rus/docs/V2400034348); act 21 dated 2024-04-30, recorded revision 2024-04-30.
  Scope (original): Пункт 3: Инструкция распространяется на расчётно-кассовые отделения вместимостью 200 и более человек. Не применять автоматически к обычной кассе: вместимость и применимость по видео не устанавливаются.

## Interpretation constraints (original)

- normative_text и official_transition_text содержат официальный текст с нормализованными пробелами; video_rules содержат аналитические условия проверки, а не цитаты закона.
- Проверять применимость нормы, все условия и исключения до вывода о событии.
- Для правил с assessment_mode=requires_external_confirmation не делать вывод о возможном нарушении без недостающих подтверждений.
- Не распознавать и не идентифицировать лица. Не определять полномочия и служебную роль по внешности.
- Отсутствие действия в одном кадре не доказывает его невыполнения. Для бездействия нужен полный релевантный интервал и достаточная видимость.
- Не выдумывать нормативы, номера пунктов, факты, таймкоды и доказательства. Не генерировать изображения или base64 вместо реальных кадров.
- Не устанавливать подлинность денег, точные суммы, цифровые подписи, проводки или принадлежность денег без доступных доказательств.
- Не считать подозрительное действие доказанным хищением или виной сотрудника.
- Проверенные нормативы относятся к указанной дате. Будущие редакции и завершившиеся переходные положения нельзя применять к другой дате без актуальной нормативной базы.
- Упоминание другого нормативного акта внутри цитаты не означает, что его содержание приведено или проверено в этом справочнике. Не придумывать недостающие требования по такой ссылке.

## Transitional provisions

[Постановление Правления Национального Банка Республики Казахстан от 24 декабря 2024 года № 88. Зарегистрировано в Министерстве юстиции Республики Казахстан 27 декабря 2024 года № 35552.](https://adilet.zan.kz/rus/docs/V2400035552#z44)

Exclusive end date: `2027-01-01`. Affected sections: `NBK_14_P11`, `NBK_14_P13`, `NBK_14_P14`, `NBK_14_P16`.

На дату проверки применяется временная редакция пункта 5 №88, уже отражённая в извлечённом тексте №14. Не добавлять до 01.01.2027 будущие требования к классу устойчивости двери, стеклу по ГОСТ, вентиляции и отсутствию отверстий между стеклом и транспортирующим устройством.

## Rule-by-rule extraction

### CASH_PERSONAL_STORAGE

Возможное хранение личных вещей или личных денег в кассовом узле

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P12](https://adilet.zan.kz/rus/docs/V1900019680#z81).

**Applicability conditions (original):**

- Подтверждено, что место является кассовым узлом, а предмет принадлежит работнику и хранится там.

**Required observable actions/evidence (original):**

- Видна последовательность помещения и оставления личного предмета на хранение в кассовом узле.

**Exceptions and insufficient-evidence cases (original):**

- Наблюдается разрешённый пунктом 12 внос лекарственных средств или медицинских изделий; сам внос не является нарушением.
- Наблюдается разрешённый пунктом 12 внос средства связи, для которого подтверждено отсутствие функции фото- и видеофиксации.
- Неизвестная принадлежность предмета или служебное назначение.
- Краткое перемещение предмета само по себе не доказывает хранение.

**Unobservable or external limits (original):**

- По внешнему виду телефона нельзя надёжно установить его функции; медицинский предмет нельзя считать запрещённым.
- Не устанавливать принадлежность денег работнику только по движению руки к карману.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_ACCEPT_COUNT

Возможное отсутствие пересчёта при приёме наличных

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P19](https://adilet.zan.kz/rus/docs/V1900019680#z104), [NBK_231_P20](https://adilet.zan.kz/rus/docs/V1900019680#z111), [NBK_231_P6](https://adilet.zan.kz/rus/docs/V1900019680#z72), [NBK_231_P41](https://adilet.zan.kz/rus/docs/V1900019680#z162), [NBK_231_P57](https://adilet.zan.kz/rus/docs/V1900019680#z196), [NBK_231_P58](https://adilet.zan.kz/rus/docs/V1900019680#z197).

**Applicability conditions (original):**

- Установлена операция приёма наличных от клиента, а не выдача или передача инкассатором.
- Виден весь этап приёма и пересчёта.

**Required observable actions/evidence (original):**

- Видна передача наличных клиентом и дальнейшие действия кассира.
- Проверяется полистный пересчёт банкнот или пересчёт монет по кружкам, включая использование оборудования.

**Exceptions and insufficient-evidence cases (original):**

- Машинный пересчёт допустим; отсутствие ручного пересчёта само по себе не является нарушением.
- Пересчёт мог происходить вне фрагмента или за перекрытием.
- Приём через автоматическое устройство требует отдельной оценки.

**Unobservable or external limits (original):**

- Количество банкнот и корректность итоговой суммы обычно не устанавливаются по общему плану.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_CLIENT_PRESENCE

Возможный приём или пересчёт наличных без присутствия клиента

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P20](https://adilet.zan.kz/rus/docs/V1900019680#z111).

**Applicability conditions (original):**

- Подтверждён приём наличных от клиента.
- Зона клиента и кассира видны в течение операции.

**Required observable actions/evidence (original):**

- Приём или пересчёт выполняется в период, когда отсутствие клиента в соответствующей зоне действительно наблюдаемо.

**Exceptions and insufficient-evidence cases (original):**

- Клиент находится за границей изображения: отсутствие на кадре не равно отсутствию в помещении.
- Неизвестно, относятся ли пересчитываемые деньги к обслуживаемому клиенту.

**Unobservable or external limits (original):**

- Присутствие следует определять по зоне обслуживания без идентификации лица.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_PREVIOUS_MONEY

Возможное оставление ранее принятых денег вне устройства хранения

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P20](https://adilet.zan.kz/rus/docs/V1900019680#z111).

**Applicability conditions (original):**

- Установлено, что деньги относятся к завершённой предыдущей операции, а не к текущему клиенту.

**Required observable actions/evidence (original):**

- Прослеживается завершение предыдущей операции и оставление её наличных на рабочей поверхности при следующей операции.

**Exceptions and insufficient-evidence cases (original):**

- Деньги относятся к текущему клиенту.
- Происхождение пачки не прослеживается.
- Устройство хранения не видно полностью.

**Unobservable or external limits (original):**

- Не утверждать, что замок не заперт, если виден только закрытый ящик; требуется наблюдение механизма или состояния.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_SEPARATE_RECEIPTS

Возможное смешение приёма наличных по нескольким приходным документам

Category: `cash_operations`. Assessment mode: `requires_external_confirmation`.

Legal basis: [NBK_231_P21](https://adilet.zan.kz/rus/docs/V1900019680#z113).

**Applicability conditions (original):**

- Достоверно известно наличие нескольких приходных кассовых документов у одного клиента.

**Required observable actions/evidence (original):**

- Видна последовательность, позволяющая проверить раздельный приём по каждому документу.

**Exceptions and insufficient-evidence cases (original):**

- Несколько листов бумаги не доказывают несколько приходных документов.
- Количество и назначение документов неизвестны.

**Unobservable or external limits (original):**

- Без читаемых документов или подтверждённых метаданных вывод обычно невозможен.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_LEAVE_WORKPLACE

Возможный уход кассира до окончания обслуживания клиента

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P22](https://adilet.zan.kz/rus/docs/V1900019680#z114).

**Applicability conditions (original):**

- Применяется этап обслуживания клиента по приёму наличных из параграфа 2.
- Виден незавершённый процесс обслуживания и границы рабочего места.

**Required observable actions/evidence (original):**

- Кассир покидает рабочее место в ходе наблюдаемой незавершённой операции.

**Exceptions and insufficient-evidence cases (original):**

- Обслуживание уже завершено.
- Кассир переместился внутри рабочего места к оборудованию.
- Граница рабочего места или завершение операции не видны.

**Unobservable or external limits (original):**

- Клиент, продолжающий стоять у окна, сам по себе не доказывает незавершённость операции.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_RECEIPT_DOCUMENT

Возможная невыдача подтверждающего документа при приёме наличных

Category: `cash_operations`. Assessment mode: `requires_external_confirmation`.

Legal basis: [NBK_231_P19](https://adilet.zan.kz/rus/docs/V1900019680#z104), [NBK_231_P10_1](https://adilet.zan.kz/rus/docs/V1900019680#z457).

**Applicability conditions (original):**

- Установлено завершение операции приёма.
- Известно, что подтверждение должно выдаваться наблюдаемым способом.

**Required observable actions/evidence (original):**

- На полном интервале завершения операции проверяется передача клиенту подтверждающего документа.

**Exceptions and insufficient-evidence cases (original):**

- Возможна выдача цифрового документа.
- Бумага или передача документа неразборчивы.
- Запись обрывается до завершения.

**Unobservable or external limits (original):**

- Отсутствие бумажной квитанции на видео не доказывает отсутствие цифрового документа.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_ISSUE_COUNT

Возможный пропуск обязательного пересчёта при выдаче наличных

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P31](https://adilet.zan.kz/rus/docs/V1900019680#z144), [NBK_231_P32](https://adilet.zan.kz/rus/docs/V1900019680#z150), [NBK_231_P41](https://adilet.zan.kz/rus/docs/V1900019680#z162).

**Applicability conditions (original):**

- Подтверждена выдача наличных клиенту.
- Виден этап подготовки, пересчёта и выдачи.

**Required observable actions/evidence (original):**

- Проверяется пересчёт подготовленных денег в присутствии клиента.
- Отдельно учитывается состояние упаковки и вид выдаваемой суммы.

**Exceptions and insufficient-evidence cases (original):**

- По пункту 32 допускается выдача полных, неполных пачек и отдельных корешков по обозначенной сумме без полистного пересчёта при неповреждённой упаковке.
- Полистный пересчёт банкнот из неполных или повреждённых корешков не следует путать с допустимой выдачей упакованных денег.
- Монеты в полной упаковке выдаются по надписям; для неполных мешков требуется пересчёт по кружкам.
- Упаковка или предшествующий пересчёт не видны.

**Unobservable or external limits (original):**

- Нельзя помечать любую выдачу пачки без ручного пересчёта как нарушение.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_ISSUE_DOCUMENT

Возможная невыдача подтверждающего документа при выдаче наличных

Category: `cash_operations`. Assessment mode: `requires_external_confirmation`.

Legal basis: [NBK_231_P31](https://adilet.zan.kz/rus/docs/V1900019680#z144), [NBK_231_P10_1](https://adilet.zan.kz/rus/docs/V1900019680#z457).

**Applicability conditions (original):**

- Подтверждено завершение выдачи наличных.
- Известен способ предоставления документа.

**Required observable actions/evidence (original):**

- Виден весь этап завершения и взаимодействия с клиентом.

**Exceptions and insufficient-evidence cases (original):**

- Документ мог быть цифровым.
- Подтверждение передано вне поля зрения.
- Операция не завершена.

**Unobservable or external limits (original):**

- Цифровое подтверждение и подписи в системе нельзя проверить по общему плану.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_EXCHANGE_ORDER

Возможное нарушение последовательности размена после выдачи

Category: `cash_operations`. Assessment mode: `requires_external_confirmation`.

Legal basis: [NBK_231_P34](https://adilet.zan.kz/rus/docs/V1900019680#z153).

**Applicability conditions (original):**

- Установлено, что клиент попросил обменять подготовленные к выдаче деньги на другой номинал.

**Required observable actions/evidence (original):**

- Проверяется, была ли сначала выполнена выдача подготовленных денег, а затем обмен.

**Exceptions and insufficient-evidence cases (original):**

- Видеозапись без звука не позволяет установить содержание просьбы клиента.
- Это первоначальная подготовка суммы или самостоятельная операция обмена.

**Unobservable or external limits (original):**

- Без подтверждённого контекста просьбы применять нельзя.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_COUNT_SUPERVISION

Возможное отсутствие контролирующего работника при обработке наличных

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P42](https://adilet.zan.kz/rus/docs/V1900019680#z163), [NBK_231_P2](https://adilet.zan.kz/rus/docs/V1900019680#z43).

**Applicability conditions (original):**

- Установлена самостоятельная операция пересчёта, сортировки или упаковки.
- Это не приём и не выдача наличных клиенту.
- Видна вся зона возможного присутствия контролирующего работника.

**Required observable actions/evidence (original):**

- На протяжении операции наблюдается отсутствие второго работника в зоне контроля.

**Exceptions and insufficient-evidence cases (original):**

- Пункт 42 прямо исключает операции приёма и выдачи наличных клиенту.
- Контролирующий работник может находиться вне кадра.
- Личность и служебная роль второго человека неизвестны.

**Unobservable or external limits (original):**

- Внешность и одежда не подтверждают должностные полномочия.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_SORT_SEPARATE

Возможное смешение разных категорий наличных при упаковке

Category: `cash_operations`. Assessment mode: `requires_external_confirmation`.

Legal basis: [NBK_231_P43](https://adilet.zan.kz/rus/docs/V1900019680#z164), [NBK_231_P45](https://adilet.zan.kz/rus/docs/V1900019680#z168).

**Applicability conditions (original):**

- Номиналы или категории годности достоверно различимы.
- Видна именно итоговая упаковка, а не временная подготовка.

**Required observable actions/evidence (original):**

- Деньги разных установленных категорий годности помещаются в одну итоговую упаковку вопреки пункту 45.

**Exceptions and insufficient-evidence cases (original):**

- Степень износа неразличима.
- Нельзя установить, изъяты ли деньги из обращения.
- Временное нахождение рядом не является совместной упаковкой.

**Unobservable or external limits (original):**

- Подлинность и платежность не определять по цвету или качеству видео.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_BANKNOTE_PACKAGING

Возможные признаки несоблюдения упаковки банкнот

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P46](https://adilet.zan.kz/rus/docs/V1900019680#z169), [NBK_231_P47](https://adilet.zan.kz/rus/docs/V1900019680#z171), [NBK_231_P48](https://adilet.zan.kz/rus/docs/V1900019680#z174), [NBK_231_P49](https://adilet.zan.kz/rus/docs/V1900019680#z176), [NBK_231_P50](https://adilet.zan.kz/rus/docs/V1900019680#z179).

**Applicability conditions (original):**

- Установлен тип упаковки: корешок, полная или неполная пачка, сборная пачка.
- Видна завершённая упаковка достаточно крупно.

**Required observable actions/evidence (original):**

- Проверяются наличие бандероли, накладок и способ закрытия именно для установленного типа упаковки.

**Exceptions and insufficient-evidence cases (original):**

- Упаковка ещё не закончена.
- Допустимая полиэтиленовая упаковка на вакуумном упаковщике не требует приписывать ей шпагатную пломбу.
- В пункте 49 предусмотрено исключение к реквизитам общей бандероли.
- Количество листов и реквизиты не читаются.

**Unobservable or external limits (original):**

- По общему плану нельзя надёжно проверить ровно 100 или 1000 листов, оттиск кода и подпись.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_COIN_PACKAGING

Возможные признаки несоблюдения упаковки монет

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P51](https://adilet.zan.kz/rus/docs/V1900019680#z503), [NBK_231_P52](https://adilet.zan.kz/rus/docs/V1900019680#z190), [NBK_231_P53](https://adilet.zan.kz/rus/docs/V1900019680), [NBK_231_P53_1](https://adilet.zan.kz/rus/docs/V1900019680#z553).

**Applicability conditions (original):**

- Подтверждены упаковка монет и используемый способ упаковки.

**Required observable actions/evidence (original):**

- Проверяются видимые повреждения мешка и завершённость закрытия/пломбирования для соответствующего способа.

**Exceptions and insufficient-evidence cases (original):**

- Отсутствие мешка со шпагатом объясняется допустимой упаковкой в пакеты-саше и транспортные пакеты по пункту 53-1; соблюдение требований к выбранному способу оценивается отдельно.
- Нельзя требовать шпагатную пломбу от любого вида пакета.
- Работа ещё продолжается или закрытие выполнено вне кадра.

**Unobservable or external limits (original):**

- Количество монет, номинал и надписи обычно требуют читаемого крупного плана.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_CASSETTE_SEAL

Возможное отсутствие закрытия и пломбирования кассеты

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P55](https://adilet.zan.kz/rus/docs/V1900019680#z193).

**Applicability conditions (original):**

- Это подготовка кассеты для загрузки наличных в автоматическое устройство.
- Виден весь этап закрытия и передачи кассеты.

**Required observable actions/evidence (original):**

- Кассета с наличными передаётся дальше с явно наблюдаемым отсутствием закрытия или пломбирования.

**Exceptions and insufficient-evidence cases (original):**

- Тип пломбы неизвестен: банк вправе выбирать бумажные, пластиковые, наклейки и другие виды.
- Закрытие/пломба вне поля зрения.
- Пустая кассета или незавершённая подготовка.

**Unobservable or external limits (original):**

- Не считать отсутствие свинцовой пломбы доказательством отсутствия пломбирования.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_BAG_INSPECTION

Возможный пропуск проверки инкассаторской сумки кассиром

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P24](https://adilet.zan.kz/rus/docs/V1900019680#z117), [NBK_231_P93](https://adilet.zan.kz/rus/docs/V1900019680#z258), [NBK_231_P94](https://adilet.zan.kz/rus/docs/V1900019680#z263).

**Applicability conditions (original):**

- Кассир банка принимает сумку от подразделения инкассации или инкассаторской организации.
- Виден весь этап приёма сумки и область пломбы.

**Required observable actions/evidence (original):**

- Проверяется наблюдаемое обследование целостности сумки и правильности пломбирования.

**Exceptions and insufficient-evidence cases (original):**

- Нужная сторона сумки закрыта от камеры.
- Краткий визуальный осмотр может не сопровождаться выраженным движением.
- Отсутствие привычной пломбы объясняется допустимым альтернативным устройством закрытия; целостность сумки и соответствующего устройства всё равно подлежат проверке.

**Unobservable or external limits (original):**

- Правильность оттиска и сравнение с образцом нельзя установить без читаемых изображений.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_DAMAGED_BAG

Возможное несоблюдение порядка работы с повреждённой сумкой

Category: `cash_operations`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P25](https://adilet.zan.kz/rus/docs/V1900019680#z123), [NBK_231_P24](https://adilet.zan.kz/rus/docs/V1900019680#z117).

**Applicability conditions (original):**

- Повреждение целостности сумки достоверно видно до её вскрытия.
- Виден полный последующий этап обработки.

**Required observable actions/evidence (original):**

- Проверяются вскрытие и пересчёт в присутствии работника инкассации.

**Exceptions and insufficient-evidence cases (original):**

- Складка, тень или шов ошибочно приняты за повреждение.
- Не видно всей зоны присутствия инкассатора.
- Неизвестно, обнаружено ли повреждение кассиром.

**Unobservable or external limits (original):**

- Акт недостачи, итог пересчёта и расследование причин требуют документов.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### COLLECTION_BAG_INSPECTION

Возможный пропуск проверки сумки инкассатором

Category: `collection`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P107](https://adilet.zan.kz/rus/docs/V1900019680#z298), [NBK_231_P108](https://adilet.zan.kz/rus/docs/V1900019680#z303), [NBK_231_P94](https://adilet.zan.kz/rus/docs/V1900019680#z263).

**Applicability conditions (original):**

- Установлено, кто принимает сумку: инкассатор-сборщик от клиента либо старший бригады от сборщика.

**Required observable actions/evidence (original):**

- Проверяется целостность сумки и пломбирования на полном этапе приёма.

**Exceptions and insufficient-evidence cases (original):**

- Видимость не позволяет оценить осмотр.
- Используется допустимое альтернативное приспособление: нельзя требовать конкретный вид пломбы, но целостность упаковки и закрытия всё равно проверяется.

**Unobservable or external limits (original):**

- Правильность документов и тождественность оттиска образцу не устанавливаются по общему плану.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### COLLECTION_PACKAGE_DEFECT

Возможный приём дефектной упаковки при перевозке

Category: `collection`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P112](https://adilet.zan.kz/rus/docs/V1900019680#z309), [NBK_231_P25](https://adilet.zan.kz/rus/docs/V1900019680#z123).

**Applicability conditions (original):**

- Подтверждён приём ценностей для перевозки по соответствующему параграфу.
- Дефект упаковки достоверно виден и сохраняется при приёме.

**Required observable actions/evidence (original):**

- Видны проверка и последующее принятие упаковки с дефектом.

**Exceptions and insufficient-evidence cases (original):**

- Это приём повреждённой сумки кассиром по пункту 25 — там предусмотрено вскрытие и пересчёт.
- Дефект неразличим или исправлен до приёма.

**Unobservable or external limits (original):**

- Нельзя переносить запрет пункта 112 на любую операцию с повреждённой сумкой.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### COLLECTION_PACK_IN_PRESENCE

Возможная упаковка для перевозки без клиента-отправителя

Category: `collection`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_231_P113](https://adilet.zan.kz/rus/docs/V1900019680#z311).

**Applicability conditions (original):**

- Это приём для перевозки по параграфу 3, а не обычная выдача сумки.
- Зона присутствия отправителя полностью видна.

**Required observable actions/evidence (original):**

- После проверки бригада упаковывает ценности без наблюдаемого присутствия отправителя.

**Exceptions and insufficient-evidence cases (original):**

- Отправитель может находиться вне кадра.
- Установить контекст перевозки невозможно.

**Unobservable or external limits (original):**

- Подписи, реквизиты ярлыка и полномочия отправителя требуют дополнительных сведений.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### COLLECTION_PROTECTION

Возможное отсутствие средств индивидуальной защиты инкассаторов

Category: `collection`. Assessment mode: `requires_external_confirmation`.

Legal basis: [NBK_231_P103](https://adilet.zan.kz/rus/docs/V1900019680#z284).

**Applicability conditions (original):**

- Подтверждено, что люди выполняют обязанности бригады инкассации.

**Required observable actions/evidence (original):**

- Можно отметить наличие или отсутствие видимых касок, бронежилетов и экипировки.

**Exceptions and insufficient-evidence cases (original):**

- Одежда может скрывать бронежилет.
- Средства выданы, но не видны в кадре.
- Роль людей неизвестна.

**Unobservable or external limits (original):**

- Пункт говорит об обеспечении экипировкой; отсутствие видимой экипировки не доказывает необеспечение.
- По видео не проверять разрешение на оружие.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### COLLECTION_CREW_SIZE

Проверка состава бригады инкассации

Category: `collection`. Assessment mode: `requires_external_confirmation`.

Legal basis: [NBK_231_P98](https://adilet.zan.kz/rus/docs/V1900019680#z268), [NBK_231_P99](https://adilet.zan.kz/rus/docs/V1900019680#z269), [NBK_231_P127](https://adilet.zan.kz/rus/docs/V1900019680#z341).

**Applicability conditions (original):**

- Установлен точный вид операции и весь состав бригады, включая водителя.

**Required observable actions/evidence (original):**

- Считается состав всей бригады на полном интервале, а не только люди у кассового окна.

**Exceptions and insufficient-evidence cases (original):**

- Пункт 99 допускает бригаду из двух человек в предусмотренных случаях.
- Сопровождение по пункту 127 имеет отдельное требование.
- Водитель или старший бригады вне кадра.

**Unobservable or external limits (original):**

- Нельзя требовать присутствия трёх инкассаторов непосредственно у кассы.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### VAULT_ACCESS

Возможный вход в кладовую или сейфовую комнату без требуемого сопровождения

Category: `storage`. Assessment mode: `requires_external_confirmation`.

Legal basis: [NBK_231_P75](https://adilet.zan.kz/rus/docs/V1900019680#z221).

**Applicability conditions (original):**

- Помещение достоверно является кладовой/сейфовой комнатой.
- Имеются подтверждённые сведения о ролях и основаниях доступа.

**Required observable actions/evidence (original):**

- Виден вход и сопровождение либо его отсутствие.

**Exceptions and insufficient-evidence cases (original):**

- Роли и распорядительный документ неизвестны.
- Человек сам является ответственным за сохранность наличных.
- Сопровождающий вне кадра.

**Unobservable or external limits (original):**

- По одежде и внешности нельзя объявлять человека посторонним или неуполномоченным.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### VAULT_KEYS

Проверка хранения ключей и кодов хранилища

Category: `storage`. Assessment mode: `requires_external_confirmation`.

Legal basis: [NBK_231_P76](https://adilet.zan.kz/rus/docs/V1900019680#z222), [NBK_231_P80](https://adilet.zan.kz/rus/docs/V1900019680#z226).

**Applicability conditions (original):**

- Установлено, что предмет является ключом от хранилища и известны ответственные лица и внутренний порядок.

**Required observable actions/evidence (original):**

- Наблюдается хранение или передача подтверждённых ключей.

**Exceptions and insufficient-evidence cases (original):**

- Назначение ключа, полномочия или внутренний порядок неизвестны.

**Unobservable or external limits (original):**

- Нельзя из общего требования самостоятельно вывести универсальный запрет оставлять любой ключ на столе.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_AREA_ISOLATION

Возможные признаки отсутствия изоляции кассового узла

Category: `premises`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_14_P8](https://adilet.zan.kz/rus/docs/V2000020075#z49).

**Applicability conditions (original):**

- Установлены границы всего кассового узла и все его входы.

**Required observable actions/evidence (original):**

- Видны конфигурация служебного входа и состояние устройств закрывания.

**Exceptions and insufficient-evidence cases (original):**

- Дополнительный закрывающийся вход для погрузочно-разгрузочных работ допускается.
- Камера охватывает только часть помещений.

**Unobservable or external limits (original):**

- Требование о закрывающемся входе не равно прямому требованию всегда держать дверь закрытой.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_DOOR

Проверка отдельного входа и запираемой двери кассы

Category: `premises`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_14_P16](https://adilet.zan.kz/rus/docs/V2000020075), [NBK_14_P18](https://adilet.zan.kz/rus/docs/V2000020075#z79).

**Applicability conditions (original):**

- Помещение является кассой; не применяется исключение пункта 18.

**Required observable actions/evidence (original):**

- Видны отдельный вход и наличие двери с возможностью закрытия на замок.

**Exceptions and insufficient-evidence cases (original):**

- Дверь находится за пределами кадра.
- Временно открытая дверь не доказывает отсутствия замка.
- Операции выполняются исключительно автоматическими устройствами по пункту 18.

**Unobservable or external limits (original):**

- Не выводить из требования к устройству двери запрет на любое её открытие.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_TRANSFER_DEVICE

Проверка транспортирующего устройства кассового окна

Category: `premises`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_14_P16](https://adilet.zan.kz/rus/docs/V2000020075), [NBK_14_P18](https://adilet.zan.kz/rus/docs/V2000020075#z79).

**Applicability conditions (original):**

- Полностью видна передняя часть оборудованной кассы.

**Required observable actions/evidence (original):**

- Проверяется наличие специального транспортирующего устройства и видимое состояние механизма.

**Exceptions and insufficient-evidence cases (original):**

- Механизм или его внутренняя часть не видны.
- Применяется исключение пункта 18 для операций с автоматическими устройствами.

**Unobservable or external limits (original):**

- Не подменять требование об оборудовании универсальным утверждением, что любое движение руки через окно незаконно.
- Запрет отверстий из будущей редакции не применять до окончания переходного периода.
- Пункт 16 допускает дополнительный инкассаторский бункер в нижней передней части кассы; его наличие само по себе не отменяет требование о транспортирующем устройстве.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### CASH_PARTITIONS

Проверка перегородок между кассами

Category: `premises`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_14_P16](https://adilet.zan.kz/rus/docs/V2000020075), [NBK_14_P18](https://adilet.zan.kz/rus/docs/V2000020075#z79).

**Applicability conditions (original):**

- Видны соседние кассы и граница между рабочими местами.

**Required observable actions/evidence (original):**

- Наблюдается наличие или явное отсутствие разделяющей перегородки между кассами.

**Exceptions and insufficient-evidence cases (original):**

- В кадре одна касса.
- Прозрачная перегородка плохо различима.
- Граница соседних касс вне кадра.
- Наличие помещения кассы не требуется по пункту 18 при операциях с автоматическими устройствами.

**Unobservable or external limits (original):**

- Из отсутствия непрозрачной стены нельзя выводить отсутствие перегородки.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### COLLECTION_LOADING_ACCESS

Возможное отсутствие ограничения доступа к погрузке ценностей

Category: `premises`. Assessment mode: `visual_candidate`.

Legal basis: [NBK_14_P21](https://adilet.zan.kz/rus/docs/V2000020075#z90).

**Applicability conditions (original):**

- Подтверждён бокс или погрузочно-разгрузочная площадка для инкассации.
- Известно, что наблюдаемое лицо является третьим лицом.

**Required observable actions/evidence (original):**

- Наблюдается возможность доступа или просмотра погрузки третьими лицами.

**Exceptions and insufficient-evidence cases (original):**

- Роли лиц неизвестны.
- Камера не показывает периметр и ограничения доступа.

**Unobservable or external limits (original):**

- Не относить клиентов и сотрудников к посторонним автоматически.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### SECURITY_GUARD_PRESENCE

Проверка организации охраны помещений

Category: `premises`. Assessment mode: `requires_external_confirmation`.

Legal basis: [NBK_14_P3](https://adilet.zan.kz/rus/docs/V2000020075#z37), [NBK_14_P4](https://adilet.zan.kz/rus/docs/V2000020075#z40).

**Applicability conditions (original):**

- Установлены применяемая схема охраны и условия местности.

**Required observable actions/evidence (original):**

- Можно фиксировать видимое присутствие охраны и оборудования.

**Exceptions and insufficient-evidence cases (original):**

- Допускается выезд по сигналу тревоги вместо постоянного поста.
- Пост или охранник вне кадра.

**Unobservable or external limits (original):**

- Отсутствие охранника в кадре не является достаточным основанием.
- Договор и подключение сигнализации требуют документов.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

### ANTITERROR_ACCESS

Проверка пропускного режима для применимых крупных отделений

Category: `antiterror`. Assessment mode: `requires_external_confirmation`.

Legal basis: [ARRFR_21_P3](https://adilet.zan.kz/rus/docs/V2400034348#z20), [ARRFR_21_P13](https://adilet.zan.kz/rus/docs/V2400034348#z62), [ARRFR_21_P15](https://adilet.zan.kz/rus/docs/V2400034348#z66), [ARRFR_21_P84](https://adilet.zan.kz/rus/docs/V2400034348#z230).

**Applicability conditions (original):**

- Подтверждена применимость Инструкции: отделение вместимостью 200 и более человек.
- Имеется утверждённый порядок доступа и схема зон.

**Required observable actions/evidence (original):**

- Видны проход через контролируемую границу и соблюдение подтверждённого порядка.

**Exceptions and insufficient-evidence cases (original):**

- Вместимость или применимость неизвестны.
- Порядок доступа и разрешённые лица неизвестны.

**Unobservable or external limits (original):**

- Само появление человека рядом с кассиром не доказывает несанкционированного доступа.

**Evidence requirements (original):**

- timestamps: Секунды от начала исходного видео.
- sequence_required: `true`
- absence_requires_complete_relevant_interval: `true`
- description: Описывать только наблюдаемое; указывать моменты, в которых видны подтверждающие действия.

## Explicitly incorrect inferences from the source

- Любой телефон или лекарство в кассе запрещены ? Пункт 12 содержит исключения; характеристики и принадлежность предмета нужно установить.
- Любая выдача денег без ручного пересчёта — нарушение ? Пункт 32 допускает выдачу определённых упаковок по обозначенным суммам; возможен машинный пересчёт.
- Один кассир у клиента всегда нарушает правило двух работников ? Пункт 42 исключает приём и выдачу денег клиенту.
- Отсутствие бумажного чека означает отсутствие документа ? Есть цифровые документы; по видео нельзя исключить их выдачу.
- Открытая дверь кассы автоматически нарушает №14 ? Текст устанавливает устройство входа и возможность запирания; универсальный запрет каждого открытия из него не следует.
- На видео нет охранника — охрана отсутствует ? Пункт 3 №14 допускает выезд по сигналу тревоги.
- У кассы меньше трёх инкассаторов — нарушение состава бригады ? Водитель может быть вне кадра, пункт 99 предусматривает исключения, сопровождение регулируется отдельно.
- Все требования антитеррористической Инструкции применимы к любой кассе ? Пункт 3 устанавливает вместимость 200 и более человек.
- Любое движение денег к карману доказывает хищение ? Видео само по себе не доказывает принадлежность денег, умысел и правовую квалификацию.
