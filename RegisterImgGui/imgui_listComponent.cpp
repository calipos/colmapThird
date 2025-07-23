#include "imgui_tools.h"
void listComponent(const std::string& listName, const ImVec2& wh, std::vector<std::string>& itemNames, int& pickIdxOut, bool& changedRightNow)
{
	if (ImGui::BeginListBox(listName.c_str(), wh))
	{
		int pickedIdx = -1;
		for (int n = 0; n < itemNames.size(); n++)
		{
			bool is_selected = (pickedIdx == n);
			if (ImGui::Selectable(itemNames[n].c_str(), is_selected))
			{
				if (pickedIdx != n)
				{
					pickIdxOut = n;
					changedRightNow = true;
					for (int k = 0; k < itemNames.size(); k++)
					{
						int itemNamesLength = itemNames[k].length();
						if (k == pickIdxOut)
						{
							itemNames[k][0] = '-';
						}
						else
						{
							itemNames[k][0] = ' ';
						}
					}
				}
				pickedIdx = n;
			}
			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndListBox();
	}
}