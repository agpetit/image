

#include "labelpointsbysectionimagetoolbox.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(LabelPointsBySectionImageToolBox)

int LabelPointsBySectionImageToolBox_Class = core::RegisterObject("LabelPointsBySectionImageToolBox")
.add< LabelPointsBySectionImageToolBox >()
.addLicense("LGPL")
.addAuthor("Vincent Majorczyk");

}}}


